import logging
import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from adan import Adan
from transformers.models.vit import ViTModel, ViTConfig, ViTForMaskedImageModeling
import torch.nn.functional as F 

from pretrain.data_model import BatchDict, Postfix, TestResults, ValDict
from pretrain.early_stopper import EarlyStopper

from pretrain.utils import (
    calculate_ranking_metrics,
    dataloader_factory,
    dir_checker,
    reduce_func,
    save_best_log,
    save_logs,
    save_predictions,
    save_test_predictions
)
from pretrain.utils import Contrastive_Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
import wandb

logger: logging.Logger = logging.getLogger()  # The logger used to log output



class TrainModule:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.state = "initializing"
        self.best_model_path: str = None
        self.save_pretrained_weights_path = self.config['pretrain']['save_path']
        self.max_len = self.config['pretrain']['max_len']

        self.vit_config = ViTConfig(image_size=self.max_len, patch_size=self.config['pretrain']['patch_size'], num_channels=self.config['pretrain']['num_channels'], encoder_stride=self.config['pretrain']['encoder_stride'])
        self.model = ViTForMaskedImageModeling(self.vit_config)
        self.model.to(self.config["device"])

        self.postfix: Postfix = {}
        self.accumulation_steps = self.config['pretrain']['accumulation_step']

        self.optimizer = self.configure_optimizers()
        self.scheduler = CosineAnnealingLR(self.optimizer, eta_min=3e-5, T_max=5000)
        if self.config["device"] != "cpu":
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.config["train"]["mixed_precision"])

        # Initialize W&B
        wandb.init(
            project="vit_pretrain",  # Replace with your project name
            config=self.config['pretrain']
        )

    def pipeline(self) -> None:
        self.config["val"]["output_dir"] = dir_checker(self.config["val"]["output_dir"])

        if self.config["train"]["model_ckpt"] is not None:
            self.model.load_state_dict(torch.load(self.config["train"]["model_ckpt"]), strict=False)
            logger.info(f'Model loaded from checkpoint: {self.config["train"]["model_ckpt"]}')

        self.t_loader = dataloader_factory(config=self.config, data_split="train")
        self.v_loader = dataloader_factory(config=self.config, data_split="val")

        self.state = "running"

        self.pbar = trange(
            self.config["train"]["epochs"], disable=(not self.config["progress_bar"]), position=0, leave=True
        )
        for epoch in self.pbar:
            if self.state in ["early_stopped", "interrupted", "finished"]:
                return

            self.postfix["Epoch"] = epoch
            self.pbar.set_postfix(self.postfix)

            try:
                self.train_procedure()
                self.save_pretrained_weights(epoch)
                self.t_loader = dataloader_factory(config=self.config, data_split="train")
            except KeyboardInterrupt:
                logger.warning("\nKeyboard Interrupt detected. Attempting graceful shutdown...")
                self.state = "interrupted"
            except Exception as err:
                raise err

            if self.state == "interrupted":
                self.validation_procedure()
                self.pbar.set_postfix(
                    {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}}
                )

        self.state = "finished"
        wandb.finish()  # Finish W&B session

    def train_procedure(self) -> None:
        self.model.train()
        train_loss_list = []

        self.max_len = self.t_loader.dataset.max_len
        for step, batch in tqdm(
            enumerate(self.t_loader),
            total=len(self.t_loader),
            disable=(not self.config["progress_bar"]),
            position=2,
            leave=False,
        ):
            train_step = self.training_step(batch, step)
            self.postfix["train_loss_step"] = float(f"{train_step['train_loss_step']:.3f}")
            train_loss_list.append(train_step["train_loss_step"])

            self.pbar.set_postfix(
                {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}}
            )

            # Log training loss to W&B
            if step % self.config["train"]["log_steps"] == 0:
                wandb.log({
                    "train_loss_step": train_step["train_loss_step"],
                    "step": step,
                    "epoch": self.postfix["Epoch"]
                })
                save_logs(
                    dict(
                        epoch=self.postfix["Epoch"],
                        seq_len=self.max_len,
                        step=step,
                        train_loss_step=f"{train_step['train_loss_step']:.3f}",
                    ),
                    output_dir=self.config["val"]["output_dir"],
                    name="log_steps",
                )

        train_loss = torch.tensor(train_loss_list)
        self.postfix["train_loss"] = train_loss.mean().item()

        # Log average train loss for the epoch
        wandb.log({"train_loss_epoch": self.postfix["train_loss"], "epoch": self.postfix["Epoch"]})

        self.validation_procedure()
        self.overfit_check()
        self.pbar.set_postfix({k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}})

    def training_step(self, batch: BatchDict, batch_idx: int) -> Dict[str, float]:
        with torch.autocast(
            device_type=self.config["device"].split(":")[0], enabled=self.config["train"]["mixed_precision"]
        ):
            reconstructed_spectrogram = self.model.forward(batch["mask"].unsqueeze(1).to(self.config["device"]))
            loss = F.huber_loss(batch["spectrogram"].unsqueeze(1).to(self.config["device"]), reconstructed_spectrogram['reconstruction'])

        loss = loss / self.accumulation_steps  # Scale loss for accumulation

        if self.config["device"] != "cpu":
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        if (batch_idx + 1) % self.accumulation_steps == 0:
            self.scheduler.step()

        return {"train_loss_step": loss.item() * self.accumulation_steps}

    def validation_procedure(self) -> None:
        self.model.eval()
        val_loss = []
        for batch in tqdm(self.v_loader, disable=(not self.config["progress_bar"]), position=1, leave=False):
            val_res = self.validation_step(batch)
            val_loss.append(val_res.item())

        avg_val_loss = np.mean(np.array(val_loss))

        # Log validation loss to W&B
        wandb.log({"val_loss_epoch": avg_val_loss, "epoch": self.postfix["Epoch"]})

        logger.info(
            f"\n{' Validation Results ':=^84}\n"
            + f"\n{avg_val_loss}"
            + f"\n{' End of Validation ':=^84}\n"
        )
        self.model.train()

    def validation_step(self, batch: BatchDict) -> torch.Tensor:
        features = self.model(batch["mask"].unsqueeze(1).to(self.config["device"]))
        return F.huber_loss(batch["spectrogram"].unsqueeze(1).to(self.config["device"]), features['reconstruction'])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adan(self.model.parameters(), lr=self.config["train"]["learning_rate"], betas=(0.98, 0.92, 0.99), weight_decay=0.15)
        return optimizer

    def save_pretrained_weights(self, epoch):
        os.makedirs(f"{self.save_pretrained_weights_path}/vit_{epoch}", exist_ok=True)

        self.model.save_pretrained(f"{self.save_pretrained_weights_path}/vit_{epoch}")

        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, os.path.join(f"{self.save_pretrained_weights_path}/vit_{epoch}", "optimizer_scheduler_state.pth"))

        print(f"Model, optimizer, and scheduler states saved at {self.save_pretrained_weights_path}/vit_{epoch}")
