import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import SwiGLU


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample (when applied in main path of residual blocks).
       This is the same as the DropConnect in EfficientNet, etc.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output
    

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise convolution
        self.norm = nn.GroupNorm(1, dim)  # заменили LayerNorm на GroupNorm для корректной обработки 4D-тензора
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise convolution 1 (expand)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # pointwise convolution 2 (project)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        x = self.norm(x)  # GroupNorm теперь работает корректно с 4D-тензором
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C) для работы со следующими слоями
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # возвращаем к виду (B, C, H, W)
        
        x = input + self.drop_path(x)
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, num_classes) -> None:
        super().__init__()
        self.swiglu = SwiGLU(in_features = embed_dim, hidden_features = 4*embed_dim, out_features=num_classes,bias=True, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0])  # аналогичная замена LayerNorm на GroupNorm
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.GroupNorm(1, dims[i]),  # аналогичная замена LayerNorm на GroupNorm
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.GroupNorm(1, dims[-1])  # финальная замена LayerNorm на GroupNorm
        # self.head = nn.Linear(dims[-1], num_classes)
        self.mlp = MLP(dims[-1], num_classes)

        self.head = nn.Conv2d(dims[-1], in_chans, kernel_size=1)  # выход для замаскированных патчей


    def forward_features(self, x):
        x = x.unsqueeze(1)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        x = self.pool(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        f_c = self.norm(x[:, :, None, None]).squeeze(-1).squeeze(-1)

        cls = self.mlp(x)
        return {'f_t':x, 'f_c':f_c, 'cls': cls}
