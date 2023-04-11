import torch
import torch.nn as nn


class Shit():
    def __init__(self) -> None:
        pass


class DirectionalAttention(nn.Module):
    def __init__(self, planes: int, midshape: tuple) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.Tensor(planes, midshape[0], midshape[0]))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        self.dim = torch.sqrt(torch.tensor(
            midshape[-1]))  # = torch.sqrt(midshape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        x = x @ x.transpose(-1, -2)
        x = x @ self.weights
        x /= self.dim
        x += identity
        x = self.bn(x)
        return x


class PlanarAttention(nn.Module):
    def __init__(self, planes: int, pooling: list = []) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.empty(planes, planes))
        nn.init.kaiming_normal_(self.weights)
        self.bn = nn.BatchNorm2d(planes)
        self.pool = nn.Identity() if len(pooling) == 0 else nn.Sequential(
            *pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        # shape is B,C,H,W
        x = self.pool(x)
        x = x.flatten(-2)
        # shape is B,C,H*W
        x = x @ x.transpose(-1, -2)
        scale = torch.var(x) / torch.var(identity)
        x = x / scale
        # shape is B,C,C
        x *= self.weights
        # shape is B,C,C
        x = torch.sum(x, dim=-1, keepdim=True)
        # shape is B,C,1
        x = x.unsqueeze(-1)
        # shape is B,C,1,1
        x = identity + x  # auto broadcast to identity
        x = self.bn(x)
        return x


class PlanarConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int | tuple = 3,
                 stride: int | tuple = 1,
                 padding: int | tuple = 1,
                 pooling: list = []) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding), nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh'))
        self.alignment = DirectionalAttention(out_channels, pooling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.alignment(x)
        return x


class DirectionalConv2d(nn.Module):
    def __init__(self,
                 midshape: tuple,
                 planes_in: int,
                 planes_out: int,
                 kernel_size: int | tuple,
                 stride: int | tuple = 1,
                 padding: int | tuple = 0,
                 groups: int = 1,
                 bias: bool = True,
                 residual: bool = False,
                 extras: list | None = None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(planes_in,
                              planes_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(planes_out)
        self.activation = nn.ReLU()
        self.extras = nn.Sequential(*extras) if extras is not None else None
        self.wm = DirectionalAttention(planes_out, midshape, residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.extras is not None:
            x = self.extras(x)
        x = self.wm(x)
        return x
