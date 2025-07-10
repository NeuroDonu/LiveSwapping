# -*- coding: utf-8 -*-
"""StyleTransferModel (128×128) перенесён в пакет `liveswapping`.

Содержит ту же архитектуру, что и в корневом `StyleTransferModel_128.py`.
"""

from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

__all__ = ["StyleTransferModel"]


class StyleTransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.style_blocks = nn.ModuleList([
            StyleBlock(1024, 1024, idx) for idx in range(6)
        ])

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.decoderPart1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.decoderPart2 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, target, source):
        target = F.pad(target, pad=(3, 3, 3, 3), mode="reflect")
        target_features = self.target_encoder(target)

        x = target_features
        for block in self.style_blocks:
            x = block(x, source)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        output = self.decoder(x)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        output = self.decoderPart1(output)
        output = F.pad(output, pad=(3, 3, 3, 3), mode="reflect")
        output = self.decoderPart2(output)
        return (output + 1) / 2


class StyleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_index: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.style1 = nn.Linear(512, 2048)
        self.style2 = nn.Linear(512, 2048)
        self.block_index = block_index

    @staticmethod
    def _normalize_conv_rms(conv):
        x = conv - torch.mean(conv, dim=[2, 3], keepdim=True)
        rms = torch.sqrt(torch.mean(x * x, dim=[2, 3], keepdim=True) + 1e-8)
        return x / rms

    def forward(self, residual, style):
        style1024 = []
        for lin in (self.style1, self.style2):
            s = lin(style)
            s = s.unsqueeze(2).unsqueeze(3)
            first, second = s[:, :1024, :, :], s[:, 1024:, :, :]
            style1024.append((first, second))

        conv1 = self._normalize_conv_rms(self.conv1(F.pad(residual, pad=(1, 1, 1, 1), mode="reflect")))
        out = torch.relu(conv1 * style1024[0][0] + style1024[0][1])
        out = F.pad(out, pad=(1, 1, 1, 1), mode="reflect")
        conv2 = self._normalize_conv_rms(self.conv2(out))
        out = conv2 * style1024[1][0] + style1024[1][1]
        return residual + out 