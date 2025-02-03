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

import flair
import pytest
import torch
from corpus import SoftLabelledTrainSetCorpus as cls
from flair.data import DataPair, Dictionary, Sentence
from flair.embeddings import TransformerDocumentEmbeddings


def test_nli_predict(tmp_path, write_nli_jsonl):
    flair.set_seed(0)
    label_dict = Dictionary()
    for l in "ABC":
        label_dict.add_item(l)
    write_nli_jsonl(["foo bar A"], "train.jsonl")
    clf = cls(tmp_path, label_dict.get_items()).make_classifier(
        TransformerDocumentEmbeddings(), label_dict
    )
    new_dp = DataPair(Sentence("foo"), Sentence("bar"))

    clf.predict(new_dp, label_name="pred soft labels")

    assert len(list(new_dp.get_labels())) == len(label_dict)
    assert all(l.score > 0 for l in new_dp.get_labels())
    assert sum(l.score for l in new_dp.get_labels()) == pytest.approx(1)


def test_bc_predict(tmp_path, write_bc_jsonl):
    flair.set_seed(0)
    label_dict = Dictionary()
    label_dict.add_item("False")
    label_dict.add_item("True")
    write_bc_jsonl(["foobar 0"], "train.jsonl")
    clf = cls(tmp_path, label_dict.get_items()).make_classifier(
        TransformerDocumentEmbeddings(), label_dict
    )
    new_dp = Sentence("foobar")

    clf.predict(new_dp, label_name="pred soft labels")

    assert len(list(new_dp.get_labels())) == len(label_dict)
    assert all(l.score > 0 for l in new_dp.get_labels())
    assert sum(l.score for l in new_dp.get_labels()) == pytest.approx(1)


def test_correct_multilabel_loss(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foobar B;1"], "train.jsonl")
    label_dict = Dictionary()
    label_dict.add_item("B")
    clf = cls(tmp_path, label_dict.get_items()).make_classifier(
        TransformerDocumentEmbeddings(), label_dict
    )

    assert isinstance(clf.loss_function, torch.nn.BCEWithLogitsLoss)
