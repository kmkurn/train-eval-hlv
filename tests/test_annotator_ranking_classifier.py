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
import pytest
import torch
from corpus import AnnotatorRankingScoredTrainSetCorpus as cls
from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerDocumentEmbeddings


def test_correct_multilabel_loss(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A;1", "bar B;2"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    label_dict.add_item("C")
    corpus = cls(tmp_path, labels=label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    sent1 = Sentence("foo")
    sent1.add_metadata("weight", 7.7)
    sent2 = Sentence("bar")
    sent2.add_metadata("weight", 3.3)

    loss, count = clf.forward_loss([sent1, sent2])

    assert count == 6
    assert loss.item() == pytest.approx(-3 * math.log(0.5) * (7.7 + 3.3))
