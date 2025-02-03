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
from peft import LoraConfig
from run_training import TransformerDocumentEmbeddingsWithLoRA


@pytest.mark.parametrize("save_lora_only", [False, True])
def test_save_load_state_dict(save_lora_only):
    t = TransformerDocumentEmbeddingsWithLoRA(LoraConfig(), save_lora_only)
    name2param_1 = dict(t.model.named_parameters())
    t.model.load_state_dict(t.model.state_dict())
    name2param_2 = dict(t.model.named_parameters())

    assert set(name2param_1.keys()) == set(name2param_2.keys())
    for name, param in name2param_1.items():
        torch.testing.assert_close(param, name2param_2[name])
