# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn

from tsn.model import registry


def round_width(width, multiplier, min_width=8, divisor=8):
    """Round channels the same way the official X3D implementation does."""
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    new_width = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return int(new_width)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation3D(nn.Module):

    def __init__(self, dim_in, se_ratio):
        super().__init__()
        dim_fc = round_width(dim_in, se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(dim_fc, dim_in, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class X3DStem(nn.Module):

    def __init__(self, dim_in, dim_out, temp_kernel=5):
        super().__init__()
        self.conv_xy = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False,
        )
        self.conv = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=(temp_kernel, 1, 1),
            stride=(1, 1, 1),
            padding=(temp_kernel // 2, 0, 0),
            groups=dim_out,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(dim_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_xy(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VideoModelStem(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.pathway0_stem = X3DStem(dim_in, dim_out)

    def forward(self, x):
        return self.pathway0_stem(x)


class X3DTransform(nn.Module):

    def __init__(self, dim_in, dim_out, dim_inner, stride, block_idx, se_ratio=0.0625):
        super().__init__()
        self.a = nn.Conv3d(dim_in, dim_inner, kernel_size=1, bias=False)
        self.a_bn = nn.BatchNorm3d(dim_inner, eps=1e-5, momentum=0.1)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            kernel_size=(3, 3, 3),
            stride=(1, stride, stride),
            padding=(1, 1, 1),
            groups=dim_inner,
            bias=False,
        )
        self.b_bn = nn.BatchNorm3d(dim_inner, eps=1e-5, momentum=0.1)
        if se_ratio > 0.0 and (block_idx + 1) % 2:
            self.se = SqueezeExcitation3D(dim_inner, se_ratio)
        self.b_relu = Swish()

        self.c = nn.Conv3d(dim_inner, dim_out, kernel_size=1, bias=False)
        self.c_bn = nn.BatchNorm3d(dim_out, eps=1e-5, momentum=0.1)

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x)
        if hasattr(self, "se"):
            x = self.se(x)
        x = self.b_relu(x)
        x = self.c(x)
        x = self.c_bn(x)
        return x


class X3DResBlock(nn.Module):

    def __init__(self, dim_in, dim_out, dim_inner, stride, block_idx):
        super().__init__()
        self.branch2 = X3DTransform(dim_in, dim_out, dim_inner, stride, block_idx)
        if dim_in != dim_out or stride != 1:
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=(1, stride, stride),
                bias=False,
            )
            self.branch1_bn = nn.BatchNorm3d(dim_out, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if hasattr(self, "branch1"):
            identity = self.branch1(x)
            identity = self.branch1_bn(identity)
        x = self.branch2(x)
        x = x + identity
        x = self.relu(x)
        return x


class X3DStage(nn.Module):

    def __init__(self, stage_index, dim_in, dim_out, dim_inner, depth, stride):
        super().__init__()
        for block_idx in range(depth):
            block = X3DResBlock(
                dim_in=dim_in if block_idx == 0 else dim_out,
                dim_out=dim_out,
                dim_inner=dim_inner,
                stride=stride if block_idx == 0 else 1,
                block_idx=block_idx,
            )
            self.add_module(f"pathway0_res{block_idx}", block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class OfficialX3DHead(nn.Module):

    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.conv_5 = nn.Conv3d(192, 432, kernel_size=1, bias=False)
        self.conv_5_bn = nn.BatchNorm3d(432, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool3d(kernel_size=(13, 5, 5), stride=1)
        self.lin_5 = nn.Conv3d(432, 2048, kernel_size=1, bias=False)
        self.lin_5_relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.projection = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        x = self.conv_5(x)
        x = self.conv_5_bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.lin_5(x)
        x = self.lin_5_relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.projection(x)
        x = x.mean(dim=(1, 2, 3))
        return x


@registry.RECOGNIZER.register("OfficialX3DRecognizer")
class OfficialX3DRecognizer(nn.Module):
    """X3D-S recognizer aligned with the official SlowFast checkpoint layout."""

    def __init__(self, cfg):
        super().__init__()
        num_classes = cfg.MODEL.HEAD.NUM_CLASSES
        dropout_rate = cfg.MODEL.HEAD.DROPOUT

        self.s1 = VideoModelStem(dim_in=3, dim_out=24)
        self.s2 = X3DStage(2, dim_in=24, dim_out=24, dim_inner=54, depth=3, stride=2)
        self.s3 = X3DStage(3, dim_in=24, dim_out=48, dim_inner=108, depth=5, stride=2)
        self.s4 = X3DStage(4, dim_in=48, dim_out=96, dim_inner=216, depth=11, stride=2)
        self.s5 = X3DStage(5, dim_in=96, dim_out=192, dim_inner=432, depth=7, stride=2)
        self.head = OfficialX3DHead(num_classes=num_classes, dropout_rate=dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, imgs):
        assert len(imgs.shape) == 5
        x = self.s1(imgs)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        probs = self.head(x)
        return {"probs": probs}
