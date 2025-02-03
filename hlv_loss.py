# Copyright 2025 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal
from warnings import warn

import torch
import torch.nn as nn


class SoftMicroF1Loss(nn.Module):
    def __init__(
        self,
        use_torch_minimum: bool = True,
        activation: Literal["sigmoid", "softmax"] = "sigmoid",
    ) -> None:
        super().__init__()
        self.__use_torch_minimum = use_torch_minimum
        self.__activation = activation

    def forward(self, logits, targets):
        if (targets > 1).any():
            warn("Some targets are unexpectedly greater than one")
        if (targets < 0).any():
            warn("Some targets are unexpectedly smaller than zero")
        preds = (
            logits.softmax(dim=-1)
            if self.__activation == "softmax"
            else logits.sigmoid()
        )
        if self.__use_torch_minimum:
            min_fn = torch.minimum
        else:
            min_fn = minimum
        loss = 1 - (2 * min_fn(preds, targets).sum() / (preds + targets).sum())
        if loss.item() < 0 or loss.item() > 1:
            warn(f"Loss is expected to be between 0 and 1 but got {loss.item()}")
        return loss

    def extra_repr(self) -> str:
        return f"use_torch_minimum={self.__use_torch_minimum!r}, activation={self.__activation!r}"


class JSDLoss(nn.Module):
    def forward(self, logits, targets):
        preds = logits.softmax(dim=-1)
        mean = 0.5 * (targets + preds)
        trg_div_mean = torch.clamp_min(targets / mean, 1e-9)
        prd_div_mean = torch.clamp_min(preds / mean, 1e-9)
        kl_trg_mean = (targets * trg_div_mean.log2()).sum(dim=-1)
        kl_prd_mean = (preds * prd_div_mean.log2()).sum(dim=-1)
        loss = 0.5 * (kl_trg_mean + kl_prd_mean).mean()
        return loss


class MultilabelJSDLoss(nn.Module):
    def forward(self, logits, targets):
        preds = logits.sigmoid()
        targets = torch.stack([targets, 1 - targets], dim=-1)
        preds = torch.stack([preds, 1 - preds], dim=-1)
        mean = 0.5 * (targets + preds)
        trg_div_mean = torch.clamp_min(targets / mean, 1e-9)
        prd_div_mean = torch.clamp_min(preds / mean, 1e-9)
        kl_trg_mean = (targets * trg_div_mean.log2()).sum(dim=-1)
        kl_prd_mean = (preds * prd_div_mean.log2()).sum(dim=-1)
        loss = 0.5 * (kl_trg_mean + kl_prd_mean).mean()
        if loss.item() < 0 or loss.item() > 1:
            warn(f"Loss is expected to be between 0 and 1 but got {loss.item()}")
        return loss


class SoftMacroF1Loss(nn.Module):
    def __init__(
        self,
        activation: Literal["sigmoid", "softmax"] = "sigmoid",
    ) -> None:
        super().__init__()
        self.__activation = activation

    def forward(self, logits, targets):
        if (targets > 1).any():
            warn("Some targets are unexpectedly greater than one")
        if (targets < 0).any():
            warn("Some targets are unexpectedly smaller than zero")
        preds = (
            logits.softmax(dim=-1)
            if self.__activation == "softmax"
            else logits.sigmoid()
        )
        loss = (
            1
            - (
                2
                * torch.minimum(preds, targets).sum(dim=0)
                / (preds + targets).sum(dim=0)
            ).mean()
        )
        if loss.item() < 0 or loss.item() > 1:
            warn(f"Loss is expected to be between 0 and 1 but got {loss.item()}")
        return loss

    def extra_repr(self) -> str:
        return f"activation={self.__activation!r}"


class MinimumFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, y):
        return torch.where(x < y, x, y)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_y = None

        grad_x = grad_output * (x < y).float()
        grad_y = grad_output * (y < x).float()

        return grad_x, grad_y


minimum = MinimumFunction.apply
