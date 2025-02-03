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
from corpus import SoftLabelledAndMajorityVotedTrainSetCorpus as cls
from flair.data import DataPair, Dictionary, Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from flair.nn import Classifier
from flair.trainers import ModelTrainer


def test_nli_embeddings_are_shared(tmp_path, mocker, write_nli_jsonl):
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    embeddings = TransformerDocumentEmbeddings()
    spy_emb_fwd = mocker.spy(embeddings, "forward")
    write_nli_jsonl(["foo bar A"], "train.jsonl")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(embeddings, label_dict)
    spy_soft_dec_fwd = mocker.spy(clf.tasks[corpus.soft_label_type].decoder, "forward")
    spy_hard_dec_fwd = mocker.spy(clf.tasks[corpus.hard_label_type].decoder, "forward")
    dp = DataPair(Sentence("foo"), Sentence("bar"))
    dp.add_label(corpus.soft_label_type, "A", score=0.3)
    dp.add_label(corpus.soft_label_type, "B", score=0.7)
    dp.add_label(corpus.hard_label_type, "B", score=0.7)
    dp.add_label("multitask_id", corpus.soft_label_type)
    dp.add_label("multitask_id", corpus.hard_label_type)

    clf.forward_loss([dp])

    assert clf.tasks[corpus.soft_label_type].embeddings is embeddings
    assert clf.tasks[corpus.hard_label_type].embeddings is embeddings
    assert spy_soft_dec_fwd.call_args.args[0].cpu().numpy() == pytest.approx(
        spy_emb_fwd.spy_return["document_embeddings"].cpu().numpy()
    )
    assert spy_hard_dec_fwd.call_args.args[0].cpu().numpy() == pytest.approx(
        spy_emb_fwd.spy_return["document_embeddings"].cpu().numpy()
    )


def test_bc_embeddings_are_shared(tmp_path, mocker, write_bc_jsonl):
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("False")
    label_dict.add_item("True")
    embeddings = TransformerDocumentEmbeddings()
    spy_emb_fwd = mocker.spy(embeddings, "forward")
    write_bc_jsonl(["foobar 0"], "train.jsonl")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(embeddings, label_dict)
    spy_soft_dec_fwd = mocker.spy(clf.tasks[corpus.soft_label_type].decoder, "forward")
    spy_hard_dec_fwd = mocker.spy(clf.tasks[corpus.hard_label_type].decoder, "forward")
    dp = Sentence("foobar")
    dp.add_label(corpus.soft_label_type, "False", score=0.3)
    dp.add_label(corpus.soft_label_type, "True", score=0.7)
    dp.add_label(corpus.hard_label_type, "True", score=0.7)
    dp.add_label("multitask_id", corpus.soft_label_type)
    dp.add_label("multitask_id", corpus.hard_label_type)

    clf.forward_loss([dp])

    assert clf.tasks[corpus.soft_label_type].embeddings is embeddings
    assert clf.tasks[corpus.hard_label_type].embeddings is embeddings
    assert spy_soft_dec_fwd.call_args.args[0].cpu().numpy() == pytest.approx(
        spy_emb_fwd.spy_return["document_embeddings"].cpu().numpy()
    )
    assert spy_hard_dec_fwd.call_args.args[0].cpu().numpy() == pytest.approx(
        spy_emb_fwd.spy_return["document_embeddings"].cpu().numpy()
    )


def test_correct_multilabel_loss(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foobar B;1"], "train.jsonl")
    label_dict = Dictionary()
    label_dict.add_item("B")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)

    assert isinstance(
        clf.tasks[corpus.soft_label_type].loss_function, torch.nn.BCEWithLogitsLoss
    )
    assert isinstance(
        clf.tasks[corpus.hard_label_type].loss_function, torch.nn.BCEWithLogitsLoss
    )


@pytest.mark.slow
def test_nli_train_and_predict(tmp_path, mocker, write_nli_jsonl):
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    write_nli_jsonl(["foo bar A"] * 7 + ["foo bar B"] * 3, "train.jsonl")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    spy_clf_fwd_loss = mocker.spy(clf, "forward_loss")
    trainer = ModelTrainer(clf, corpus)

    trainer.fine_tune(tmp_path / "artifacts", max_epochs=1)
    clf = Classifier.load(tmp_path / "artifacts" / "final-model.pt")
    dp = DataPair(Sentence("bar"), Sentence("baz"))
    clf.predict(dp)
    fwd_loss_dp_arg = spy_clf_fwd_loss.call_args.args[0][0]

    assert len(spy_clf_fwd_loss.call_args.args[0]) == 1
    assert fwd_loss_dp_arg.first.text == "foo"
    assert fwd_loss_dp_arg.second.text == "bar"
    assert {
        l.value: l.score for l in fwd_loss_dp_arg.get_labels(corpus.soft_label_type)
    } == pytest.approx({"A": 0.7, "B": 0.3})
    assert fwd_loss_dp_arg.get_label(corpus.hard_label_type).value == "A"
    assert dp.get_labels(corpus.soft_label_type)
    assert {l.value for l in dp.get_labels(corpus.soft_label_type)} == set(
        label_dict.get_items()
    )
    assert sum(l.score for l in dp.get_labels(corpus.soft_label_type)) == pytest.approx(
        1
    )


@pytest.mark.slow
def test_bc_train_and_predict(tmp_path, mocker, write_bc_jsonl):
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("False")
    label_dict.add_item("True")
    write_bc_jsonl(["foobar 0"] * 7 + ["foobar 1"] * 3, "train.jsonl")
    corpus = cls(tmp_path, label_dict.get_items())
    clf = corpus.make_classifier(TransformerDocumentEmbeddings(), label_dict)
    spy_clf_fwd_loss = mocker.spy(clf, "forward_loss")
    trainer = ModelTrainer(clf, corpus)

    trainer.fine_tune(tmp_path / "artifacts", max_epochs=1)
    clf = Classifier.load(tmp_path / "artifacts" / "final-model.pt")
    dp = Sentence("barbaz")
    clf.predict(dp)
    fwd_loss_dp_arg = spy_clf_fwd_loss.call_args.args[0][0]

    assert len(spy_clf_fwd_loss.call_args.args[0]) == 1
    assert fwd_loss_dp_arg.get_metadata("orig_text") == "foobar"
    assert {
        l.value: l.score for l in fwd_loss_dp_arg.get_labels(corpus.soft_label_type)
    } == pytest.approx({"False": 0.7, "True": 0.3})
    assert fwd_loss_dp_arg.get_label(corpus.hard_label_type).value == "False"
    assert dp.get_labels(corpus.soft_label_type)
    assert {l.value for l in dp.get_labels(corpus.soft_label_type)} == set(
        label_dict.get_items()
    )
    assert sum(l.score for l in dp.get_labels(corpus.soft_label_type)) == pytest.approx(
        1
    )
