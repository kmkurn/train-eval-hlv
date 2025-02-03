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
import math
import pytest
import torch
from corpus import AnnotationCollectedTrainSetCorpus as cls
from corpus import SoftLabelledTrainSetCorpus as soft_cls
from flair.data import DataPair, Dictionary, Sentence
from flair.embeddings import TransformerDocumentEmbeddings


def test_correct_nli_loss(tmp_path, write_nli_jsonl):
    write_nli_jsonl(["foo bar E", "foo bar N", "foo bar C", "foo bar E"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("E")
    label_dict.add_item("N")
    label_dict.add_item("C")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.constant_(clf.decoder.bias[label_dict.get_idx_for_item("E")], -2)
    torch.nn.init.constant_(clf.decoder.bias[label_dict.get_idx_for_item("C")], -1)

    expected = math.log(math.exp(-2) + math.exp(0) + math.exp(-1))

    loss, _ = clf.forward_loss(list(corpus.train))

    assert loss.item() == pytest.approx(expected)


def test_correct_binary_classification_loss(tmp_path, write_bc_jsonl):
    write_bc_jsonl(["foo 1", "foo 0", "foo 1"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("False")
    label_dict.add_item("True")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.constant_(clf.decoder.bias[label_dict.get_idx_for_item("False")], -2)
    torch.nn.init.constant_(clf.decoder.bias[label_dict.get_idx_for_item("True")], -1)

    expected = 1 + math.log(math.exp(-2) + math.exp(-1))

    loss, _ = clf.forward_loss(list(corpus.train))

    assert loss.item() == pytest.approx(expected)


def test_correct_multilabel_loss(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A;", "foo A B;", "foo B;"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.constant_(clf.decoder.bias[label_dict.get_idx_for_item("A")], -2)
    torch.nn.init.constant_(clf.decoder.bias[label_dict.get_idx_for_item("B")], -1)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    expected = -(math.log(1 - sigmoid(-2)) + math.log(sigmoid(-1)))

    loss, _ = clf.forward_loss(list(corpus.train))

    assert loss.item() == pytest.approx(expected)


@pytest.mark.parametrize("aggfunc", ["min", "max", "mean"])
def test_correct_nli_batch_loss(tmp_path, write_nli_jsonl, aggfunc):
    flair.set_seed(0)
    write_nli_jsonl(["foo bar E"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("E")
    label_dict.add_item("N")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    clf.loss_function.aggfunc = aggfunc
    s1 = DataPair(Sentence("foo"), Sentence("bar"))
    s1.add_label("annotation 0", "E")
    s1.add_label("annotation 1", "N")
    s1.add_metadata("num_annotations", 2)
    s2 = DataPair(Sentence("baz"), Sentence("quux"))
    s2.add_label("annotation 0", "E")
    s2.add_metadata("num_annotations", 1)
    expected = clf.forward_loss([s1])[0] + clf.forward_loss([s2])[0]

    batch_loss, _ = clf.forward_loss([s1, s2])

    assert batch_loss.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("aggfunc", ["min", "max", "mean"])
def test_correct_bc_batch_loss(tmp_path, write_bc_jsonl, aggfunc):
    flair.set_seed(0)
    write_bc_jsonl(["foo 0"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("False")
    label_dict.add_item("True")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    clf.loss_function.aggfunc = aggfunc
    s1 = Sentence("foo")
    s1.add_label("annotation 0", "False")
    s1.add_label("annotation 1", "True")
    s1.add_metadata("num_annotations", 2)
    s2 = Sentence("bar")
    s2.add_label("annotation 0", "False")
    s2.add_metadata("num_annotations", 1)
    expected = clf.forward_loss([s1])[0] + clf.forward_loss([s2])[0]

    batch_loss, _ = clf.forward_loss([s1, s2])

    assert batch_loss.item() == pytest.approx(expected.item())


@pytest.mark.parametrize("aggfunc", ["min", "max", "mean"])
def test_correct_multilabel_batch_loss(tmp_path, write_mullab_jsonl, aggfunc):
    flair.set_seed(0)
    write_mullab_jsonl(["foo A;"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    clf.loss_function.aggfunc = aggfunc
    s1 = Sentence("foo")
    s1.add_label("annotation 0", "A")
    s1.add_label("annotation 1", "B")
    s1.add_metadata("num_annotations", 2)
    s2 = Sentence("bar")
    s2.add_label("annotation 0", "A")
    s2.add_metadata("num_annotations", 1)
    expected = clf.forward_loss([s1])[0] + clf.forward_loss([s2])[0]

    batch_loss, _ = clf.forward_loss([s1, s2])

    assert batch_loss.item() == pytest.approx(expected.item())


def test_nli_mean_equal_soft_classifier(tmp_path, write_nli_jsonl):
    flair.set_seed(0)
    write_nli_jsonl(
        ["foo bar E", "foo bar N", "foo bar C", "foo bar N", "baz quux E"],
        "train.jsonl",
    )
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("E")
    label_dict.add_item("N")
    label_dict.add_item("C")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    clf.loss_function.aggfunc = "mean"
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.uniform_(clf.decoder.bias)
    soft_corpus = soft_cls(tmp_path, label_dict.get_items())
    soft_clf = soft_corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    for p in soft_clf.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        soft_clf.decoder.bias.data = clf.decoder.bias.data.clone()
    expected = soft_clf.forward_loss(soft_corpus.train)[0]

    loss, _ = clf.forward_loss(corpus.train)

    assert loss.item() == pytest.approx(expected.item())


def test_bc_mean_equal_soft_classifier(tmp_path, write_bc_jsonl):
    flair.set_seed(0)
    write_bc_jsonl(["foo 1", "foo 0", "foo 1", "bar 0"], "train.jsonl")
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("False")
    label_dict.add_item("True")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    clf.loss_function.aggfunc = "mean"
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.uniform_(clf.decoder.bias)
    soft_corpus = soft_cls(tmp_path, label_dict.get_items())
    soft_clf = soft_corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    for p in soft_clf.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        soft_clf.decoder.bias.data = clf.decoder.bias.data.clone()
    expected = soft_clf.forward_loss(soft_corpus.train)[0]

    loss, _ = clf.forward_loss(corpus.train)

    assert loss.item() == pytest.approx(expected.item())


def test_multilabel_mean_equal_soft_classifier(tmp_path, write_mullab_jsonl):
    flair.set_seed(0)
    write_mullab_jsonl(
        ["foo A;", "foo A B;", "foo;", "foo A B;", "bar A;"], "train.jsonl"
    )
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    clf.loss_function.aggfunc = "mean"
    for p in clf.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.uniform_(clf.decoder.bias)
    soft_corpus = soft_cls(tmp_path, label_dict.get_items())
    soft_clf = soft_corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    for p in soft_clf.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        soft_clf.decoder.bias.data = clf.decoder.bias.data.clone()
    expected = soft_clf.forward_loss(soft_corpus.train)[0]

    loss, _ = clf.forward_loss(corpus.train)

    assert loss.item() == pytest.approx(expected.item())
