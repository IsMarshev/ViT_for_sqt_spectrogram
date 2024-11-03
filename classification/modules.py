import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit import ViTModel, ViTConfig, ViTForImageClassification
from xformers.ops import SwiGLU



class MLP(nn.Module):
    def __init__(self, embed_dim, num_classes) -> None:
        super().__init__()
        self.swiglu = SwiGLU(in_features = embed_dim, hidden_features = 4*embed_dim, out_features=num_classes,bias=True, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)

class ViTForImageClassificationCustom(nn.Module):
    def __init__(self, num_classes):
        super(ViTForImageClassificationCustom, self).__init__()

        self.vit = ViTModel.from_pretrained("E:/yandex_cup/models_chkpnt/pretrain/vit_6")
        self.norm = nn.LayerNorm(self.vit.config.hidden_size)
        self.classifier = MLP(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        norm_pooled_output = self.norm(pooled_output)
        logits = self.classifier(norm_pooled_output)
        return {'f_t': pooled_output, 'f_c': norm_pooled_output, 'cls': logits}


