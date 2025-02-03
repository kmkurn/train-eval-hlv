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
from corpus import AnnotatorLabelledTrainSetCorpus as cls
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

    loss, count = clf.forward_loss(list(corpus.train))

    assert count == 6
    assert loss.item() == pytest.approx(-count * math.log(0.5))


def test_predict_multilabel(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A;1"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    clf = cls(tmp_path, labels=label_dict.get_items()).make_classifier(
        TransformerDocumentEmbeddings(), label_dict
    )
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.ones_(clf.decoder.bias)
    sent = Sentence("bar")

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    clf.predict(sent)

    assert {l.value: l.score for l in sent.get_labels()} == {
        l: pytest.approx(sigmoid(1)) for l in label_dict.get_items()
    }


def test_predict_multilabel_averages_votes(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A;1", "foo A;2", "foo A;3"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    clf = cls(tmp_path, labels=label_dict.get_items()).make_classifier(
        TransformerDocumentEmbeddings(), label_dict
    )
    clf.averages_probs = False
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    # TODO don't access "private" attribute _ann_dict
    torch.nn.init.ones_(clf.decoder.bias[clf._ann_dict.get_idx_for_item("1")])
    torch.nn.init.ones_(
        clf.decoder.bias[
            clf._ann_dict.get_idx_for_item("2"), label_dict.get_idx_for_item("A")
        ]
    )
    sent = Sentence("bar")

    clf.predict(sent)

    assert {l.value: l.score for l in sent.get_labels()} == {"A": pytest.approx(2 / 3)}


def test_predict_multilabel_return_loss(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A;1"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    clf = cls(tmp_path, labels=label_dict.get_items()).make_classifier(
        TransformerDocumentEmbeddings(), label_dict
    )
    sent = Sentence("bar")
    expected, _ = clf.forward_loss([sent])

    loss, _ = clf.predict(sent, return_loss=True)

    assert loss.item() == pytest.approx(expected.item())
