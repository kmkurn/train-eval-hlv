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

import math

import numpy as np
import pytest
import torch
from hlv_loss import JSDLoss, MultilabelJSDLoss


def test_multiclass_correct(mocker):
    torch.manual_seed(0)
    p0, p1 = 0.2, 0.3
    p2 = 1 - (p0 + p1)
    q0, q1 = 0.1, 0.7
    q2 = 1 - (q0 + q1)
    targets = torch.tensor([[p0, p1, p2]])
    preds = torch.tensor([[q0, q1, q2]])
    m0 = (p0 + q0) / 2
    m1 = (p1 + q1) / 2
    m2 = (p2 + q2) / 2
    kl_p_m = p0 * math.log2(p0 / m0) + p1 * math.log2(p1 / m1) + p2 * math.log2(p2 / m2)
    kl_q_m = q0 * math.log2(q0 / m0) + q1 * math.log2(q1 / m1) + q2 * math.log2(q2 / m2)
    expected = (kl_p_m + kl_q_m) / 2
    logits = torch.rand([1, 3])
    fake_softmax = mocker.patch.object(
        logits, "softmax", autospec=True, return_value=preds
    )

    loss = JSDLoss()(logits, targets)

    torch.testing.assert_close(loss, torch.tensor(expected))
    fake_softmax.assert_called_once_with(dim=-1)


def test_multilabel_correct():
    p0, p1 = 0.2, 0.3
    q0, q1 = 0.9, 0.4
    targets = torch.tensor([[p0, p1]])
    preds = torch.tensor([[q0, q1]])
    m0 = np.array([(p0 + q0), (1 - p0 + 1 - q0)]) / 2
    m1 = np.array([(p1 + q1), (1 - p1 + 1 - q1)]) / 2
    kl_p0 = p0 * math.log2(p0 / m0[0]) + (1 - p0) * math.log2((1 - p0) / m0[1])
    kl_p1 = p1 * math.log2(p1 / m1[0]) + (1 - p1) * math.log2((1 - p1) / m1[1])
    kl_q0 = q0 * math.log2(q0 / m0[0]) + (1 - q0) * math.log2((1 - q0) / m0[1])
    kl_q1 = q1 * math.log2(q1 / m1[0]) + (1 - q1) * math.log2((1 - q1) / m1[1])
    jsd0 = (kl_p0 + kl_q0) / 2
    jsd1 = (kl_p1 + kl_q1) / 2
    expected = (jsd0 + jsd1) / 2
    logits = (preds / (1 - preds)).log()

    loss = MultilabelJSDLoss()(logits, targets)

    torch.testing.assert_close(loss, torch.tensor(expected))


@pytest.mark.parametrize("invalid_value", [0, 1])
def test_multilabel_zero_target(invalid_value):
    torch.manual_seed(0)
    targets = torch.rand([1, 2])
    targets[0, 0] = invalid_value
    preds = torch.rand([1, 2])
    logits = (preds / (1 - preds)).log().requires_grad_(True)

    loss = MultilabelJSDLoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss).all()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("invalid_value", [-100, 100])
def test_multilabel_zero_pred(invalid_value):
    torch.manual_seed(0)
    targets = torch.rand([1, 2])
    logits = torch.rand([1, 2])
    logits[0, 0] = invalid_value
    logits.requires_grad_(True)

    loss = MultilabelJSDLoss()(logits, targets)
    loss.backward()

    assert torch.isfinite(loss).all()
    assert torch.isfinite(logits.grad).all()


def test_multilabel_warn_invalid_targets():
    loss_fn = MultilabelJSDLoss()
    with pytest.warns(UserWarning):
        loss_fn(torch.tensor([[-0.03]]), torch.tensor([[1.2]]))
    with pytest.warns(UserWarning):
        loss_fn(torch.tensor([[-0.03]]), torch.tensor([[-0.2]]))
