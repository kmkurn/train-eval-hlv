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

from unittest.mock import Mock

import pytest
import torch.nn as nn
from flair.data import Corpus, DataPair, Dictionary, Sentence
from flair.datasets import DataPairCorpus, FlairDatapointDataset
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import MultitaskModel, TextPairClassifier
from flair.trainers import ModelTrainer
from peft import LoraConfig, get_peft_model


@pytest.mark.slow
def test_train_and_predict(tmp_path):
    (tmp_path / "train.tsv").write_text("foo	bar	A\nfoo	baz	B\n", "utf8")
    corpus = DataPairCorpus(tmp_path, train_file="train.tsv", label_type="class")
    classifier = TextPairClassifier(
        TransformerDocumentEmbeddings(),
        label_type="class",
        label_dictionary=corpus.make_label_dictionary(label_type="class"),
    )
    trainer = ModelTrainer(classifier, corpus)

    trainer.fine_tune(tmp_path / "artifacts", max_epochs=1)
    classifier = TextPairClassifier.load(tmp_path / "artifacts" / "final-model.pt")
    dp = DataPair(Sentence("foo"), Sentence("bar"))
    classifier.predict(dp)
    pred = dp.get_label()

    assert isinstance(pred.value, str)
    assert isinstance(pred.score, float)
    assert 0 <= pred.score <= 1


def test_save_dictionary(tmp_path):
    d = Dictionary(add_unk=False)
    d.add_item("B")
    d.add_item("A")

    d.save(tmp_path / "dict")
    d_ = Dictionary.load_from_file(tmp_path / "dict")

    assert d == d_


def test_classifier_embeddings_are_public():
    embeddings = TransformerDocumentEmbeddings()

    clf = TextPairClassifier(
        embeddings, label_type="label", label_dictionary=Dictionary()
    )

    assert clf.embeddings is embeddings


def test_transformer_embeddings_are_torch_module():
    assert isinstance(TransformerDocumentEmbeddings(), nn.Module)


def test_embeddings_output_is_fed_into_classifier_decoder(mocker):
    label_dict = Dictionary(add_unk=False)
    label_dict.add_item("A")
    label_dict.add_item("B")
    embeddings = TransformerDocumentEmbeddings()
    spy_emb_fwd = mocker.spy(embeddings, "forward")
    clf = TextPairClassifier(
        embeddings, label_type="label", label_dictionary=label_dict
    )
    spy_clf_fwd = mocker.spy(clf.decoder, "forward")
    dp = DataPair(Sentence("foo"), Sentence("bar"))
    dp.add_label("label", "A")

    clf.forward_loss([dp])

    assert spy_clf_fwd.call_args.args[0].cpu().numpy() == pytest.approx(
        spy_emb_fwd.spy_return["document_embeddings"].cpu().numpy()
    )


@pytest.mark.slow
def test_classifier_forward_loss_called_when_finetuned(tmp_path, mocker):
    dp = DataPair(Sentence("foo"), Sentence("bar"))
    dp.add_label("label", "A")
    corpus = Corpus(FlairDatapointDataset(dp))
    clf = TextPairClassifier(
        TransformerDocumentEmbeddings(),
        label_type="label",
        label_dictionary=corpus.make_label_dictionary("label"),
    )
    spy_fwd_loss = mocker.spy(clf, "forward_loss")
    trainer = ModelTrainer(clf, corpus)

    trainer.fine_tune(tmp_path / "artifacts", max_epochs=1)

    assert spy_fwd_loss.called


@pytest.mark.slow
def test_train_multitask_model(tmp_path):
    dp = DataPair(Sentence("foo"), Sentence("bar"))
    dp.add_label("task_1", "A")
    dp.add_label("task_2", "X")
    dp.add_label("multitask_id", "TASK_1")
    dp.add_label("multitask_id", "TASK_2")
    corpus = Corpus(FlairDatapointDataset(dp))
    label_dict_1 = Dictionary(add_unk=False)
    label_dict_1.add_item("A")
    label_dict_1.add_item("B")
    label_dict_2 = Dictionary(add_unk=False)
    label_dict_2.add_item("X")
    label_dict_2.add_item("Y")
    clf = MultitaskModel(
        [
            TextPairClassifier(
                TransformerDocumentEmbeddings(),
                label_type="task_1",
                label_dictionary=label_dict_1,
            ),
            TextPairClassifier(
                TransformerDocumentEmbeddings(),
                label_type="task_2",
                label_dictionary=label_dict_2,
            ),
        ],
        task_ids=["TASK_1", "TASK_2"],
        loss_factors=[0.5, 0.5],
        use_all_tasks=True,
    )
    trainer = ModelTrainer(clf, corpus)

    trainer.fine_tune(tmp_path / "artifacts", max_epochs=1)
    clf = MultitaskModel.load(tmp_path / "artifacts" / "final-model.pt")
    new_dp = DataPair(dp.first, dp.second)
    clf.predict(new_dp)

    assert new_dp.get_label("task_1").value in label_dict_1.get_items()
    assert new_dp.get_label("task_2").value in label_dict_2.get_items()


def test_forward_multitask_model_with_shared_embeddings(tmp_path, monkeypatch):
    label_dict_1 = Dictionary(add_unk=False)
    label_dict_1.add_item("A")
    label_dict_1.add_item("B")
    label_dict_2 = Dictionary(add_unk=False)
    label_dict_2.add_item("X")
    label_dict_2.add_item("Y")
    embeddings = TransformerDocumentEmbeddings()
    mock_forward = Mock(wraps=embeddings.forward)
    monkeypatch.setattr(embeddings, "forward", mock_forward)
    clf = MultitaskModel(
        [
            TextPairClassifier(
                embeddings,
                label_type="task_1",
                label_dictionary=label_dict_1,
            ),
            TextPairClassifier(
                embeddings,
                label_type="task_2",
                label_dictionary=label_dict_2,
            ),
        ],
        task_ids=["TASK_1", "TASK_2"],
        loss_factors=[0.5, 0.5],
        use_all_tasks=True,
    )
    dp = DataPair(Sentence("foo"), Sentence("bar"))
    dp.add_label("task_1", "A")
    dp.add_label("task_2", "X")
    dp.add_label("multitask_id", "TASK_1")
    dp.add_label("multitask_id", "TASK_2")

    clf.forward_loss([dp])

    assert mock_forward.call_count == 2


def test_task_classifiers_in_multitask_model_are_public():
    clf_task1 = TextPairClassifier(
        TransformerDocumentEmbeddings(),
        label_type="task_1",
        label_dictionary=Dictionary(),
    )
    clf_task2 = TextPairClassifier(
        TransformerDocumentEmbeddings(),
        label_type="task_2",
        label_dictionary=Dictionary(),
    )
    clf = MultitaskModel(
        [clf_task1, clf_task2],
        task_ids=["TASK_1", "TASK_2"],
    )

    assert clf.tasks["TASK_1"] is clf_task1
    assert clf.tasks["TASK_2"] is clf_task2


def test_add_metadata_to_datapair():
    dp = DataPair(Sentence("foo"), Sentence("bar"))

    dp.add_metadata("annotator", "asdf")

    assert dp.get_metadata("annotator") == "asdf"


def test_len_of_text_pair():
    s = DataPair(Sentence("foo bar baz"), Sentence("quux bar"))

    assert len(s) == 5


def test_sentence_with_trailing_whitespace():
    s = Sentence("foo bar ")
    assert s.text == "foo bar"


def test_sentence_get_label_is_first():
    s = Sentence("foo")
    s.add_label("class", "A", score=0.2)
    s.add_label("class", "B", score=0.7)
    s.add_label("class", "C", score=0.1)

    assert s.get_label("class").value == "A"


def test_dictionary_get_idx():
    d1 = Dictionary()
    d2 = Dictionary(add_unk=False)
    [idx1] = d1.get_idx_for_items(["foo"])
    [idx2] = d2.get_idx_for_items(["foo"])

    assert len(d1) == 1
    assert len(d2) == 0
    assert d1.get_items() == ["<unk>"]
    assert d2.get_items() == []
    assert idx1 == 0
    assert idx2 == 0
    assert d1.get_item_for_index(idx1) == "<unk>"
    with pytest.raises(IndexError):
        d2.get_item_for_index(idx2)


def test_corpus_has_dev_and_test_sets_as_private_attributes():
    dat = FlairDatapointDataset([Sentence("foo")])
    corpus = Corpus(dat, dat, dat)
    assert corpus._dev is dat
    assert corpus._test is dat


@pytest.mark.parametrize("trg_mods", [None, "all-linear"])
def test_fewer_trainable_parameters_with_lora(trg_mods):
    emb = TransformerDocumentEmbeddings()
    n_params = sum(p.numel() for p in emb.model.parameters() if p.requires_grad)
    emb.model = get_peft_model(emb.model, LoraConfig(target_modules=trg_mods))
    n_params_lora = sum(p.numel() for p in emb.model.parameters() if p.requires_grad)

    assert n_params_lora < n_params


def test_get_labels_same_type():
    s = Sentence("foo")
    s.add_label("class", "A")
    s.add_label("class", "B")
    s.add_label("class", "A")

    assert [l.value for l in s.get_labels("class")] == ["A", "B", "A"]
