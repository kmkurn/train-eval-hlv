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

import pytest
import torch
from hlv_loss import SoftMicroF1Loss


def test_correct():
    targets = torch.tensor([[0.2, 0.3], [0.1, 0.8], [0.9, 0.7]])
    preds = torch.tensor([[0.8, 0.1], [0.3, 0.4], [0.7, 0.4]])
    soft_micro_f1 = (
        2 * (0.2 + 0.1 + 0.1 + 0.4 + 0.7 + 0.4) / (1.0 + 0.4 + 0.4 + 1.2 + 1.6 + 1.1)
    )
    logits = (preds / (1 - preds)).log()

    loss = SoftMicroF1Loss()(logits, targets)

    torch.testing.assert_close(loss, torch.tensor(1 - soft_micro_f1))


@pytest.mark.parametrize("use_torch_minimum", [False, True])
def test_differentiable_at_minimum(use_torch_minimum):
    targets = torch.rand([2, 3])
    inputs = (targets / (1 - targets)).log().requires_grad_(True)
    loss = SoftMicroF1Loss(use_torch_minimum)(inputs, targets)

    loss.backward()

    assert torch.isfinite(inputs.grad).all()


def test_warn_invalid_targets():
    loss_fn = SoftMicroF1Loss()
    with pytest.warns(UserWarning):
        loss_fn(torch.tensor([-0.03]), torch.tensor([1.2]))
    with pytest.warns(UserWarning):
        loss_fn(torch.tensor([-0.03]), torch.tensor([-0.2]))


def test_with_softmax_activation(mocker):
    torch.manual_seed(0)
    logits = torch.rand([2, 3])
    preds = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.6, 0.1]])
    fake_softmax = mocker.patch.object(
        logits, "softmax", autospec=True, return_value=preds
    )
    targets = torch.tensor([[0.2, 0.4, 0.4], [0.1, 0.1, 0.8]])
    soft_accuracy = 0.5 * (0.1 + 0.4 + 0.4 + 0.1 + 0.1 + 0.1)

    loss = SoftMicroF1Loss(activation="softmax")(logits, targets)

    torch.testing.assert_close(loss, torch.tensor(1 - soft_accuracy))
    fake_softmax.assert_called_once_with(dim=-1)
