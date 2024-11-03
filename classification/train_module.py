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

from classification.data_model import BatchDict, Postfix, TestResults, ValDict
from classification.early_stopper import EarlyStopper
from classification.modules import ViTForImageClassificationCustom
from classification.utils import (
    calculate_ranking_metrics,
    dataloader_factory,
    dir_checker,
    reduce_func,
    save_best_log,
    save_logs,
    save_predictions,
    save_test_predictions
)
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb


logger: logging.Logger = logging.getLogger()  # The logger used to log output


class TrainModule:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.state = "initializing"
        self.best_model_path: str = None
        self.num_classes = self.config["train"]["num_classes"]
        self.save_weights_path = self.config['train']['save_path']
        self.max_len = 84

        self.model = ViTForImageClassificationCustom(self.num_classes)

        self.model.to(self.config["device"])

        self.postfix: Postfix = {}

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=config["train"]["triplet_margin"])
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=config["train"]["smooth_factor"])


        self.optimizer = self.configure_optimizers()
        self.scheduler = CosineAnnealingLR(self.optimizer, eta_min=3e-6, T_max=1000)
        self.early_stop = EarlyStopper(patience=self.config["train"]["patience"])

        if self.config["device"] != "cpu":
            #self.scaler = torch.cuda.amp.GradScaler(enabled=self.config["train"]["mixed_precision"])
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.config["train"]["mixed_precision"])
        
        wandb.init(
            project="vit_classification", 
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
            except KeyboardInterrupt:
                logger.warning("\nKeyboard Interrupt detected. Attempting gracefull shutdown...")
                self.state = "interrupted"
            except Exception as err:
                raise (err)

            #'''
            if self.state == "interrupted":
                self.validation_procedure()
                self.pbar.set_postfix(
                    {k: self.postfix[k] for k in self.postfix.keys() & {"train_loss_step", "mr1", "mAP"}}
                )
            #'''

        self.state = "finished"

    def validate(self) -> None:
        self.v_loader = dataloader_factory(config=self.config, data_split="val")
        self.state = "running"
        self.validation_procedure()
        self.state = "finished"

    def test(self) -> None:
        self.test_loader = dataloader_factory(config=self.config, data_split="test")
        self.test_results: TestResults = {}

        if self.best_model_path is not None:
            self.model.load_state_dict(torch.load(self.best_model_path), strict=False)
            logger.info(f"Best model loaded from checkpoint: {self.best_model_path}")
        elif self.config["test"]["model_ckpt"] is not None:
            self.model.load_state_dict(torch.load(self.config["test"]["model_ckpt"]), strict=False)
            logger.info(f'Model loaded from checkpoint: {self.config["test"]["model_ckpt"]}')
        elif self.state == "initializing":
            logger.warning("Warning: Testing with random weights")

        self.state = "running"
        self.test_procedure()
        self.state = "finished"

    def train_procedure(self) -> None:
        self.model.train()
        train_loss_list = []
        train_cls_loss_list = []
        train_csv_loss_list = []
        
        for step, batch in tqdm(
            enumerate(self.t_loader),
            total=len(self.t_loader),
            disable=(not self.config["progress_bar"]),
            position=2,
            leave=False,
        ):
            train_step = self.training_step(batch)
            self.postfix["train_loss_step"] = float(f"{train_step['train_loss_step']:.3f}")
            train_loss_list.append(train_step["train_loss_step"])
            self.postfix["train_cls_loss_step"] = float(f"{train_step['train_cls_loss']:.3f}")
            train_cls_loss_list.append(train_step["train_cls_loss"])
            self.postfix["train_csv_loss_list_step"] = float(f"{train_step['train_csv_loss']:.3f}")
            train_csv_loss_list.append(train_step["train_csv_loss"])

            # Логгируем каждые log_steps шагов
            if step % self.config["train"]["log_steps"] == 0:
                wandb.log({
                    "train/loss": train_step["train_loss_step"],
                    "train/cls_loss": train_step["train_cls_loss"],
                    "train/csv_loss": train_step["train_csv_loss"],
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "epoch": self.postfix["Epoch"],
                })

        # После эпохи обновляем scheduler и логгируем средний лосс за эпоху
        self.scheduler.step()

        wandb.log({
            "train/epoch_loss": torch.tensor(train_loss_list).mean().item(),
            "train/epoch_cls_loss": torch.tensor(train_cls_loss_list).mean().item(),
            "train/epoch_csv_loss": torch.tensor(train_csv_loss_list).mean().item(),
        })

        self.validation_procedure()

    def training_step(self, batch: BatchDict) -> Dict[str, float]:
        with torch.autocast(
            device_type=self.config["device"].split(":")[0], enabled=self.config["train"]["mixed_precision"]
        ):
            anchor = self.model.forward(batch["anchor"].unsqueeze(1).to(self.config["device"]))
            positive = self.model.forward(batch["positive"].unsqueeze(1).to(self.config["device"]))
            negative = self.model.forward(batch["negative"].unsqueeze(1).to(self.config["device"]))
            l1 = self.triplet_loss(anchor["f_t"], positive["f_t"], negative["f_t"])
            labels = nn.functional.one_hot(batch["anchor_label"].long(), num_classes=self.num_classes)
            l2 = self.cls_loss(anchor["cls"], labels.float().to(self.config["device"]))
            loss = l1 + l2

        self.optimizer.zero_grad()
        if self.config["device"] != "cpu":
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {"train_loss_step": loss.item(), "train_csv_loss": l1.item(), "train_cls_loss": l2.item()}

    def validation_procedure(self) -> None:
        self.model.eval()
        embeddings: Dict[int, torch.Tensor] = {}
        
        for batch in tqdm(self.v_loader, disable=(not self.config["progress_bar"]), position=1, leave=False):
            val_dict = self.validation_step(batch)
            for anchor_id, triplet_embedding, embedding in zip(val_dict["anchor_id"], val_dict["f_t"], val_dict["f_c"]):
                embeddings[anchor_id] = torch.stack([triplet_embedding, embedding])

        val_outputs = self.validation_epoch_end(embeddings)

        # Логгируем метрики в wandb
        wandb.log({
            "val/mAP": self.postfix["mAP"],
            "val/mrr": self.postfix["mrr"]
        })

        self.model.train()

    def validation_epoch_end(self, outputs: Dict[int, torch.Tensor]) -> Dict[int, np.ndarray]:
        #val_loss = torch.zeros(len(outputs))
        #pos_ids = []
        #neg_ids = []
        clique_ids = []
        for k, (anchor_id, embeddings) in enumerate(outputs.items()):
            #clique_id, pos_id, neg_id = self.v_loader.dataset._triplet_sampling(anchor_id)
            #val_loss[k] = self.triplet_loss(embeddings[0], outputs[pos_id][0], outputs[neg_id][0]).item()
            #pos_ids.append(pos_id)
            #neg_ids.append(neg_id)
            clique_id = self.v_loader.dataset.version2clique.loc[anchor_id, 'clique']
            clique_ids.append(clique_id)
        #anchor_ids = np.stack(list(outputs.keys()))
        preds = torch.stack(list(outputs.values()))[:, 1]
        #self.postfix["val_loss"] = val_loss.mean().item()
        rranks, average_precisions = calculate_ranking_metrics(embeddings=preds.numpy(), cliques=clique_ids)
        self.postfix["mrr"] = rranks.mean()
        self.postfix["mAP"] = average_precisions.mean()
        return {
            #"triplet_ids": np.stack(list(zip(clique_ids, anchor_ids, pos_ids, neg_ids))),
            "rranks": rranks,
            "average_precisions": average_precisions,
        }

    def validation_step(self, batch: BatchDict) -> ValDict:
        anchor_id = batch["anchor_id"]
        features = self.model.forward(batch["anchor"].unsqueeze(1).to(self.config["device"]))

        return {
            "anchor_id": anchor_id.numpy(),
            "f_t": features["f_t"].squeeze(0).detach().cpu(),
            "f_c": features["f_c"].squeeze(0).detach().cpu(),
        }

    def test_procedure(self) -> None:
        self.model.eval()
        embeddings: Dict[str, torch.Tensor] = {}
        trackids: List[int] = []
        embeddings: List[np.array] = []
        for batch in tqdm(self.test_loader, disable=(not self.config["progress_bar"])):
            test_dict = self.validation_step(batch)
            if test_dict["f_c"].ndim == 1:
                test_dict["f_c"] = test_dict["f_c"].unsqueeze(0)
            for anchor_id, embedding in zip(test_dict["anchor_id"], test_dict["f_c"]):
                trackids.append(anchor_id)
                embeddings.append(embedding.numpy())
        predictions = []
        for chunk_result in pairwise_distances_chunked(embeddings, metric='cosine', reduce_func=reduce_func, working_memory=100):
            for query_indx, query_nearest_items in chunk_result:
                predictions.append((trackids[query_indx], [trackids[nn_indx] for nn_indx in query_nearest_items]))
        save_test_predictions(predictions, output_dir=self.config["test"]["output_dir"])

    def overfit_check(self) -> None:
        if self.early_stop(self.postfix["mAP"]):
            logger.info(f"\nValidation not improved for {self.early_stop.patience} consecutive epochs. Stopping...")
            self.state = "early_stopped"

        if self.early_stop.counter > 0:
            logger.info("\nValidation mAP was not improved")
        else:
            logger.info(f"\nMetric improved. New best score: {self.early_stop.max_validation_mAP:.3f}")
            save_best_log(self.postfix, output_dir=self.config["val"]["output_dir"])

            logger.info("Saving model...")
            epoch = self.postfix["Epoch"]
            max_secs = self.max_len
            prev_model = deepcopy(self.best_model_path)
            self.best_model_path = os.path.join(
                self.config["val"]["output_dir"], "model", f"best-model-{epoch=}-{max_secs=}.pt"
            )
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            torch.save(deepcopy(self.model.state_dict()), self.best_model_path)
            if prev_model is not None:
                os.remove(prev_model)

    def save_pretrained_weights(self, epoch):
        os.makedirs(f"{self.save_weights_path}/vit_{epoch}", exist_ok=True)

        self.model.save_pretrained(f"{self.save_weights_path}/vit_{epoch}")

        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }, os.path.join(f"{self.save_pretrained_weights_path}/vit_{epoch}", "optimizer_scheduler_state.pth"))

        print(f"Model, optimizer, and scheduler states saved at {self.save_pretrained_weights_path}/vit_{epoch}")

    def configure_optimizers(self) -> torch.optim.Optimizer:        
        optimizer = Adan(self.model.parameters(), lr = self.config["train"]["learning_rate"], betas = (0.98,0.98,0.99), weight_decay=0.05)

        return optimizer
