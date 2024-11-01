import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob)

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False):
        """Drop paths per sample (when applied in the main path of residual blocks)."""
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # Work with any shape
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binomial distribution
        output = x / keep_prob * random_tensor
        return output
    
    
class DynamicLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super(DynamicLayerNorm, self).__init__()
        self.num_channels = num_channels

    def forward(self, x):
        # x имеет размерность [batch_size, num_channels, H, W]
        # Рассчитываем H и W динамически
        _, _, H, W = x.shape
        layer_norm = nn.LayerNorm([self.num_channels, H, W]).to(x.device)
        return layer_norm(x)
    

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.norm = DynamicLayerNorm(dim)

        self.pointwise_conv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gelu = nn.GELU()


    def forward(self, x):
        residual = x.clone()
        x = self.conv(x)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        # if self.downsample is not None:
        #     residual = self.downsample(residual)
            
        out = residual + x

        out = self.gelu(out)
        return out 



class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x + eps, kernel_size=x.shape[2:], stride=1).pow(p).mean(dim=(2, 3)).pow(1. / p)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        num_channels: int = 1,  # Входные каналы, например, 1 для одноканальных данных
        num_classes: int = 39535,  # Количество классов в выходе
        depths=(3, 3, 4, 6, 3),  # Количество блоков ConvNeXt на каждом этапе
        dims=(96, 192, 384, 768),  # Размерность каналов на каждом этапе
        drop_path_rate=0.0,
        emb_dim = 3072,
        layer_scale_init_value=1e-6,
        dropout=0.1,
    ):
        super(ConvNeXt, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(num_channels, dims[0], kernel_size=7, stride=2, padding=3),
            DynamicLayerNorm(dims[0])
        )


        self.layer1 = self._make_layer(ConvNeXtBlock, blocks=3, planes=96)
        self.layer2 = self._make_layer(ConvNeXtBlock, blocks=4, planes=192)
        self.layer3 = self._make_layer(ConvNeXtBlock, blocks=6, planes=384)
        self.layer4 = self._make_layer(ConvNeXtBlock, blocks=3, planes=768)
        

        # GeM пуллинг и классификационная голова
        self.gem_pool = GeM()
        self.dropout = nn.Dropout(p=dropout)

        self.bn_fc = nn.BatchNorm1d(emb_dim)

        self.fc = nn.Linear(emb_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.fc.weight)

    def _make_layer(self, ConvNeXtBlock: ConvNeXtBlock, blocks: int, planes:int):
        
        layers = []
        # layers.append(
        #     ConvNeXtBlock(dim=self.in_channels, stride=stride, downsample=downsample, last=last)
        # )
        for _ in range(0, blocks):
            layers.append(ConvNeXtBlock(dim=planes))
        # if stride != 1:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.in_channels, kernel_size=1, stride=stride, bias=False),
        #         nn.DynamicLayerNorm(self.in_channels),
        #     )
        layers.append(nn.Conv2d(planes, planes*2, kernel_size= 2, stride = 2))

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f_t = self.layer4(x)
        
        # f_t = self.gem_pool(x)
        f_t = self.dropout(torch.flatten(f_t, start_dim=1))
        f_c = self.bn_fc(f_t)
        cls = self.fc(f_c)

        return dict(f_t=f_t, f_c=f_c, cls=cls)