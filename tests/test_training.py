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

import abc
import contextlib
import itertools
import json
import unittest.mock

import corpus as corpus_module
import flair
import numpy as np
import pytest
import run_training as run_training_module
from corpus import AnnotatorEnsembleTextClassifier as AETClassifier
from corpus import AnnotatorEnsembleTextPairClassifier as AETPClassifier
from corpus import LossAggregationTextClassifier as LATClassifier
from corpus import LossAggregationTextPairClassifier as LATPClassifier
from flair.data import Dictionary
from flair.models import TextClassifier, TextPairClassifier
from predict import predict
from run_training import train as run_train
from sacred.run import Run


@pytest.fixture
def train_and_predict(tmp_path):
    def _train_and_predict(
        method=None, use_lora=False, lora_target=None, multilabel=False
    ):
        kwargs = {}
        if method:
            kwargs["method"] = method
        train(
            tmp_path,
            tmp_path / "artifacts",
            use_lora=use_lora,
            lora_target=lora_target,
            **kwargs,
        )
        predict(
            tmp_path / "artifacts",
            tmp_path / "inputs.jsonl",
            tmp_path / "preds.jsonl",
            multilabel,
        )
        return map(
            json.loads, (tmp_path / "preds.jsonl").read_text("utf8").splitlines()
        )

    return _train_and_predict


@pytest.fixture
def mock_eval_funcs(mocker):
    @contextlib.contextmanager
    def _mock_eval_funcs(overall_metric_names, splits, labels):
        n_splits = len(splits)
        overall_metric_rvs = np.random.rand(len(overall_metric_names), n_splits)
        name2mock = {
            name: mocker.Mock(side_effect=rvs)
            for name, rvs in zip(overall_metric_names, overall_metric_rvs)
        }
        for name, mock in name2mock.items():
            mocker.patch(f"run_training.{name}", mock)
        hard_prfs_rvs = np.random.rand(n_splits, 4, 2)
        mock_hard_prfs = mocker.Mock(side_effect=hard_prfs_rvs)
        mocker.patch("run_training.hard_prfs", mock_hard_prfs)
        soft_prfs_rvs = np.random.rand(n_splits, 4, 2)
        mock_soft_prfs = mocker.Mock(side_effect=soft_prfs_rvs)
        mocker.patch("run_training.soft_prfs", mock_soft_prfs)
        mock_run = mocker.Mock(spec=Run)
        yield (
            overall_metric_rvs,
            [name2mock[name] for name in overall_metric_names]
            + [mock_hard_prfs, mock_soft_prfs],
            mock_run,
        )

        for mock in list(name2mock.values()) + [mock_hard_prfs, mock_soft_prfs]:
            assert len(mock.call_args_list) == len(splits)
            assert mock.call_args_list[splits.index("dev")].args[0] == pytest.approx(
                np.array([[1 / 3, 2 / 3]])
            )
            assert mock.call_args_list[splits.index("test")].args[0] == pytest.approx(
                np.array([[2 / 3, 1 / 3]])
            )
            for call_args in mock.call_args_list:
                assert len(call_args.args) == 2
                assert call_args.args[1].shape == call_args.args[0].shape

        call = unittest.mock.call
        for name, mock in name2mock.items():
            for i, split in enumerate(splits):
                assert (
                    call(
                        f"{split}/{name}",
                        pytest.approx(
                            overall_metric_rvs[overall_metric_names.index(name), i]
                        ),
                    )
                    in mock_run.log_scalar.call_args_list
                )
        for k, split in enumerate(splits):
            for t, mock, rvs in zip(
                "hard soft".split(),
                [mock_hard_prfs, mock_soft_prfs],
                [hard_prfs_rvs, soft_prfs_rvs],
            ):
                for i, x in enumerate("p r f1 supp".split()):
                    if x != "supp":
                        assert (
                            call(
                                f"{split}/macro/{t}_{x}",
                                pytest.approx(rvs[k, i].mean()),
                            )
                            in mock_run.log_scalar.call_args_list
                        )
                    if split == "test":
                        for j, l in enumerate(labels):
                            assert (
                                call(
                                    f"{split}/{t}_{x}/{l}", pytest.approx(rvs[k, i, j])
                                )
                                in mock_run.log_scalar.call_args_list
                            )

    return _mock_eval_funcs


@pytest.mark.slow
@pytest.mark.parametrize(
    "method", "ReL MV SL SLMV AL SmF1 JSD SMF1 LA-min LA-max LA-mean".split()
)
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_nli_ok(write_nli_jsonl, train_and_predict, method, use_lora, lora_target):
    flair.set_seed(0)
    write_nli_jsonl(["foo bar E", "baz quux N"], "train.jsonl")
    write_nli_jsonl(["foo bar", "baz quux"], "inputs.jsonl")

    preds = list(train_and_predict(method, use_lora, lora_target))

    assert len(preds) == 2
    if method == "ReL" and not use_lora:
        assert preds == [{"verdict": "E"}, {"verdict": "N"}]


@pytest.mark.parametrize(
    "method",
    "ReL MV SL SLMV AL AE AEh AR ARh SmF1 SMF1 JSD LA-min LA-max LA-mean".split(),
)
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
@pytest.mark.slow
def test_binary_classification_ok(
    write_bc_jsonl, train_and_predict, method, use_lora, lora_target
):
    flair.set_seed(0)
    write_bc_jsonl(["foo 1 A", "bar 0 B"], "train.jsonl")
    write_bc_jsonl(["foo", "bar"], "inputs.jsonl")

    preds = list(train_and_predict(method, use_lora, lora_target))

    assert len(preds) == 2
    if method == "ReL" and not use_lora:
        assert preds == [{"class": True}, {"class": False}]


@pytest.mark.slow
@pytest.mark.parametrize(
    "method",
    "ReL MV SL SLMV AE AEh AR ARh SmF1 SmF1-v2 SMF1 JSD LA-min LA-max LA-mean".split(),
)
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_multilabel_classification_ok(
    write_mullab_jsonl, train_and_predict, method, use_lora, lora_target, mocker
):
    flair.set_seed(0)
    write_mullab_jsonl(["foo;1", "bar B;2", "foobar A B;3"], "train.jsonl")
    write_mullab_jsonl(["foo", "bar", "foobar"], "inputs.jsonl")

    preds = list(train_and_predict(method, use_lora, lora_target, multilabel=True))

    assert len(preds) == 3
    for pred in preds:
        assert all(c in ("A", "B") for c in pred["classes"])
    if method == "ReL" and not use_lora:
        assert preds[0] == {"classes": []}
        assert preds[1] == {"classes": ["B"]}
        assert preds[2] == {"classes": ["A", "B"]} or preds[2] == {
            "classes": ["B", "A"]
        }


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,loss_cls_name",
    list(
        zip(
            "SmF1 SMF1 JSD".split(),
            "SoftMicroF1Loss SoftMacroF1Loss JSDLoss".split(),
        )
    ),
)
def test_nli_correct_soft_loss(
    write_nli_jsonl, mocker, tmp_path, method, loss_cls_name
):
    write_nli_jsonl(["foo bar E", "baz quux N"], "train.jsonl")
    spy_loss_cls = mocker.spy(run_training_module, loss_cls_name)
    spy_clf_cls = mocker.spy(corpus_module, "TextPairSoftClassifier")

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)

    kwargs = (
        {"activation": "softmax"}
        if loss_cls_name in ("SoftMicroF1Loss", "SoftMacroF1Loss")
        else {}
    )
    spy_loss_cls.assert_called_once_with(**kwargs)
    assert spy_clf_cls.spy_return.loss_function == spy_loss_cls.spy_return


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,loss_cls_name",
    list(
        zip(
            "SmF1 SMF1 JSD".split(),
            "SoftMicroF1Loss SoftMacroF1Loss JSDLoss".split(),
        )
    ),
)
def test_binary_classification_correct_soft_loss(
    write_bc_jsonl, mocker, tmp_path, method, loss_cls_name
):
    write_bc_jsonl(["foo 1", "bar 0"], "train.jsonl")
    spy_loss_cls = mocker.spy(run_training_module, loss_cls_name)
    spy_clf_cls = mocker.spy(corpus_module, "TextSoftClassifier")

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)

    if method != "JSD":
        spy_loss_cls.assert_called_once_with(activation="softmax")
    else:
        spy_loss_cls.assert_called_once_with()
    assert spy_clf_cls.spy_return.loss_function == spy_loss_cls.spy_return


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,loss_cls_name",
    list(
        zip(
            "SmF1 SmF1-v2 SMF1 JSD".split(),
            "SoftMicroF1Loss SoftMicroF1Loss SoftMacroF1Loss MultilabelJSDLoss".split(),
        )
    ),
)
def test_multilabel_classification_correct_soft_loss(
    write_mullab_jsonl, mocker, tmp_path, method, loss_cls_name
):
    write_mullab_jsonl(["foo;1", "bar B;2", "foobar A B;3"], "train.jsonl")
    spy_loss_cls = mocker.spy(run_training_module, loss_cls_name)
    spy_clf_cls = mocker.spy(corpus_module, "TextSoftClassifier")

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)

    if method != "SmF1-v2":
        spy_loss_cls.assert_called_once_with()
    else:
        spy_loss_cls.assert_called_once_with(use_torch_minimum=False)
    assert spy_clf_cls.spy_return.loss_function == spy_loss_cls.spy_return


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,hide_eval_sets_from_trainer",
    list(
        itertools.product(
            "ReL MV AL SL SLMV AE AEh SmF1 SMF1 JSD".split(), [False, True]
        )
    )
    + [(f"LA-{x}", True) for x in "min max mean".split()],
)
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_nli_eval_sets_exist(
    tmp_path,
    write_nli_jsonl,
    mocker,
    mock_eval_funcs,
    method,
    hide_eval_sets_from_trainer,
    use_lora,
    lora_target,
):
    write_nli_jsonl(["foo bar E 1", "baz baz N 2"], "train.jsonl")
    write_nli_jsonl(["foo bar N", "foo bar E", "foo bar N"], "dev.jsonl")
    write_nli_jsonl(["foo bar E", "foo bar N", "foo bar E "], "test.jsonl")
    overall_metric_names = "hard_acc poJSD ent_corr soft_acc".split()
    splits = "dev test".split()
    with mock_eval_funcs(overall_metric_names, splits, labels=("E", "N")) as (
        overall_metric_rvs,
        all_mocks,
        mock_run,
    ):
        test_score = train_one_epoch(
            tmp_path,
            tmp_path / "artifacts",
            method,
            hide_eval_sets_from_trainer,
            use_lora,
            lora_target,
            _run=mock_run,
        )

        assert test_score == pytest.approx(
            overall_metric_rvs[
                overall_metric_names.index("hard_acc"), splits.index("dev")
            ]
        )

        for mock in all_mocks:
            for call_args in mock.call_args_list:
                assert call_args.args[1].sum(axis=1) == pytest.approx(1.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,hide_eval_sets_from_trainer",
    list(
        itertools.product(
            "ReL MV AL SL SLMV AE AEh AR ARh SmF1 SMF1 JSD".split(), [False, True]
        )
    )
    + [(f"LA-{x}", True) for x in "min max mean".split()],
)
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_binary_classification_eval_sets_exist(
    tmp_path,
    write_bc_jsonl,
    method,
    mocker,
    mock_eval_funcs,
    hide_eval_sets_from_trainer,
    use_lora,
    lora_target,
):
    write_bc_jsonl(["foo 1 A", "baz 0 B"], "train.jsonl")
    write_bc_jsonl(["foo 0", "foo 1", "foo 0"], "dev.jsonl")
    write_bc_jsonl(["foo 1", "foo 0", "foo 1"], "test.jsonl")
    overall_metric_names = "hard_acc poJSD ent_corr soft_acc".split()
    splits = "dev test".split()
    with mock_eval_funcs(overall_metric_names, splits, labels=("True", "False")) as (
        overall_metric_rvs,
        all_mocks,
        mock_run,
    ):
        test_score = train_one_epoch(
            tmp_path,
            tmp_path / "artifacts",
            method,
            hide_eval_sets_from_trainer,
            use_lora,
            lora_target,
            _run=mock_run,
        )

        assert test_score == pytest.approx(
            overall_metric_rvs[
                overall_metric_names.index("hard_acc"), splits.index("dev")
            ]
        )

        for mock in all_mocks:
            for call_args in mock.call_args_list:
                assert call_args.args[1].sum(axis=1) == pytest.approx(1.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,hide_eval_sets_from_trainer",
    list(
        itertools.product(
            "ReL MV SL SLMV AE AEh AR ARh SmF1 SmF1-v2 SMF1 JSD".split(), [False, True]
        )
    )
    + [(f"LA-{x}", True) for x in "min max mean".split()],
)
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_multilabel_classification_eval_sets_exist(
    tmp_path,
    write_mullab_jsonl,
    method,
    mock_eval_funcs,
    hide_eval_sets_from_trainer,
    use_lora,
    lora_target,
):
    write_mullab_jsonl(["foo A;1", "baz A B;2"], "train.jsonl")
    write_mullab_jsonl(["foo;", "foo B;", "foo A B;"], "dev.jsonl")
    write_mullab_jsonl(["foo A;", "foo;", "foo A B;"], "test.jsonl")
    overall_metric_names = "hard_micro_f1 poJSD ent_corr soft_micro_f1".split()
    splits = "dev test".split()
    with mock_eval_funcs(overall_metric_names, splits, labels=("A", "B")) as (
        overall_metric_rvs,
        all_mocks,
        mock_run,
    ):
        test_score = train_one_epoch(
            tmp_path,
            tmp_path / "artifacts",
            method,
            hide_eval_sets_from_trainer,
            use_lora,
            lora_target,
            labels=("A", "B"),
            _run=mock_run,
        )

        assert test_score == pytest.approx(
            overall_metric_rvs[
                overall_metric_names.index("hard_micro_f1"), splits.index("dev")
            ]
        )
        for name, mock in zip(
            overall_metric_names + ["hard_prfs", "soft_prfs"], all_mocks
        ):
            for call_args in mock.call_args_list:
                assert ((0 <= call_args.args[1]) & (call_args.args[1] <= 1)).all()
            if name in ("poJSD", "ent_corr", "hard_prfs"):
                assert mock.call_args_list[splits.index("dev")].kwargs == {
                    "multilabel": True
                }
                assert mock.call_args_list[splits.index("test")].kwargs == {
                    "multilabel": True
                }


def test_nli_labels_given(write_nli_jsonl, mocker, tmp_path):
    write_nli_jsonl(["foo bar E"], "train.jsonl")
    write_nli_jsonl(["foo bar N"], "dev.jsonl")
    mocker.patch("run_training.ModelTrainer.fine_tune", autospec=True)

    train_one_epoch(tmp_path, tmp_path / "artifacts", labels=("N", "E"))


def test_binary_classification_labels_given(write_bc_jsonl, mocker, tmp_path):
    write_bc_jsonl(["foo 1"], "train.jsonl")
    write_bc_jsonl(["foo 0"], "dev.jsonl")
    mocker.patch("run_training.ModelTrainer.fine_tune", autospec=True)

    train_one_epoch(tmp_path, tmp_path / "artifacts", labels=("False", "True"))


def test_multilabel_classification_labels_given(write_mullab_jsonl, mocker, tmp_path):
    write_mullab_jsonl(["foo A;"], "train.jsonl")
    write_mullab_jsonl(["foo B;"], "dev.jsonl")
    mocker.patch("run_training.ModelTrainer.fine_tune", autospec=True)

    train_one_epoch(tmp_path, tmp_path / "artifacts", labels=("B", "A"))


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,corpus_cls_name",
    list(
        zip(
            "ReL MV AL SL SLMV AE AEh SmF1 SMF1 JSD LA-min LA-max LA-mean".split(),
            [
                "DCorpus",
                "MVCorpus",
                "ALCorpus",
                "SLCorpus",
                "SLMVCorpus",
                "AnnLCorpus",
                "AnnLCorpus",
                "SLCorpus",
                "SLCorpus",
                "SLCorpus",
                "ACCorpus",
                "ACCorpus",
                "ACCorpus",
            ],
        )
    ),
)
@pytest.mark.parametrize("eval_split", ["dev", "test"])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_nli_hlvs_saved(
    tmp_path,
    write_nli_jsonl,
    mocker,
    method,
    corpus_cls_name,
    eval_split,
    use_lora,
    lora_target,
):
    write_nli_jsonl(["foo bar N 1", "baz quux E 2"], "train.jsonl")
    write_nli_jsonl(
        [
            "foo bar E",
            "bar baz N",
            "foo bar N",
            "foo foo N",
            "bar bar E",
            "bar baz E",
            "bar baz E",
        ],
        f"{eval_split}.jsonl",
    )
    input2p = {  # true, pred
        ("foo", "bar"): (1 / 2, 0.9),
        ("bar", "baz"): (2 / 3, 0.3),
        ("foo", "foo"): (0 / 1, 0.6),
        ("bar", "bar"): (1 / 1, 0.2),
    }

    class FakeTextPairClassifier(FakePredictMixinABC, TextPairClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            dp.add_label(label_name, "E", get_pred_p(dp))
            dp.add_label(label_name, "N", 1 - get_pred_p(dp))

    class FakeAETPClassifier(FakePredictMixinABC, AETPClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            dp.add_label(label_name, "E", get_pred_p(dp))
            dp.add_label(label_name, "N", 1 - get_pred_p(dp))

    class FakeLATPClassifier(FakePredictMixinABC, LATPClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            dp.add_label(label_name, "E", get_pred_p(dp))
            dp.add_label(label_name, "N", 1 - get_pred_p(dp))

    def get_pred_p(dp):
        return input2p[dp.first.text, dp.second.text][1]

    mocker.patch(
        f"run_training.{corpus_cls_name}.make_classifier",
        build_fake_make_classifier(
            corpus_cls_name,
            FakeAETPClassifier,
            FakeLATPClassifier,
            FakeTextPairClassifier,
            anns=("1", "2"),
        ),
    )
    artifacts_dir = tmp_path / "artifacts"

    train_one_epoch(
        tmp_path, artifacts_dir, method, use_lora=use_lora, lora_target=lora_target
    )
    label_dict, inputs, hlv = load_artifacts(artifacts_dir, eval_split)
    true_hlv, pred_hlv = hlv

    assert len(label_dict) == 2
    assert label_dict.get_idx_for_items(("N", "E")) == [0, 1]
    for (prem, hyp), (true_p, pred_p) in input2p.items():
        idx = inputs.index({"premise": prem, "hypothesis": hyp})
        assert true_hlv[idx] == pytest.approx([1 - true_p, true_p])
        assert pred_hlv[idx] == pytest.approx([1 - pred_p, pred_p])


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,corpus_cls_name",
    list(
        zip(
            "ReL MV AL SL SLMV AE AEh AR ARh SmF1 SMF1 JSD LA-min LA-max LA-mean".split(),
            [
                "DCorpus",
                "MVCorpus",
                "ALCorpus",
                "SLCorpus",
                "SLMVCorpus",
                "AnnLCorpus",
                "AnnLCorpus",
                "ARCorpus",
                "ARCorpus",
                "SLCorpus",
                "SLCorpus",
                "SLCorpus",
                "ACCorpus",
                "ACCorpus",
                "ACCorpus",
            ],
        )
    ),
)
@pytest.mark.parametrize("eval_split", ["dev", "test"])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_binary_classification_hlvs_saved(
    tmp_path,
    write_bc_jsonl,
    mocker,
    method,
    corpus_cls_name,
    eval_split,
    use_lora,
    lora_target,
):
    write_bc_jsonl(["foobar 0 A", "bazquux 1 B"], "train.jsonl")
    write_bc_jsonl(
        [
            "foobar 1",
            "barbaz 0",
            "foobar 0",
            "foofoo 0",
            "barbar 1",
            "barbaz 1",
            "barbaz 1",
        ],
        f"{eval_split}.jsonl",
    )
    input2p = {  # true, pred
        "foobar": (1 / 2, 0.9),
        "barbaz": (2 / 3, 0.3),
        "foofoo": (0 / 1, 0.6),
        "barbar": (1 / 1, 0.2),
    }

    class FakeTextClassifier(FakePredictMixinABC, TextClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            dp.add_label(label_name, "True", get_pred_p(dp))
            dp.add_label(label_name, "False", 1 - get_pred_p(dp))

    class FakeAETClassifier(FakePredictMixinABC, AETClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            dp.add_label(label_name, "True", get_pred_p(dp))
            dp.add_label(label_name, "False", 1 - get_pred_p(dp))

    class FakeLATClassifier(FakePredictMixinABC, LATClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            dp.add_label(label_name, "True", get_pred_p(dp))
            dp.add_label(label_name, "False", 1 - get_pred_p(dp))

    def get_pred_p(dp):
        return input2p[dp.get_metadata("orig_text")][1]

    mocker.patch(
        f"run_training.{corpus_cls_name}.make_classifier",
        build_fake_make_classifier(
            corpus_cls_name,
            FakeAETClassifier,
            FakeLATClassifier,
            FakeTextClassifier,
            anns=("A", "B"),
        ),
    )
    artifacts_dir = tmp_path / "artifacts"

    train_one_epoch(
        tmp_path, artifacts_dir, method, use_lora=use_lora, lora_target=lora_target
    )
    label_dict, inputs, hlv = load_artifacts(artifacts_dir, eval_split)
    true_hlv, pred_hlv = hlv

    assert len(label_dict) == 2
    assert label_dict.get_idx_for_items(("False", "True")) == [0, 1]
    for text, (true_p, pred_p) in input2p.items():
        idx = inputs.index({"text": text})
        assert true_hlv[idx] == pytest.approx([1 - true_p, true_p])
        assert pred_hlv[idx] == pytest.approx([1 - pred_p, pred_p])


@pytest.mark.slow
@pytest.mark.parametrize(
    "method,corpus_cls_name",
    list(
        zip(
            "ReL MV SL SLMV AE AEh AR ARh SmF1 SmF1-v2 SMF1 JSD LA-min LA-max LA-mean".split(),
            "DCorpus MVCorpus SLCorpus SLMVCorpus AnnLCorpus AnnLCorpus ARCorpus ARCorpus SLCorpus SLCorpus SLCorpus SLCorpus ACCorpus ACCorpus ACCorpus".split(),
        )
    ),
)
@pytest.mark.parametrize("eval_split", ["dev", "test"])
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_multilabel_classification_hlvs_saved(
    tmp_path,
    write_mullab_jsonl,
    mocker,
    method,
    corpus_cls_name,
    eval_split,
    use_lora,
    lora_target,
):
    write_mullab_jsonl(["foobar A;1", "bazquux B;2"], "train.jsonl")
    write_mullab_jsonl(
        [
            "foobar B;",
            "barbaz A;",
            "foobar A B;",
            "foofoo A;",
            "barbar B;",
            "barbaz A B;",
            "barbaz A;",
        ],
        f"{eval_split}.jsonl",
    )
    input2label2p = {  # true, pred
        "foobar": ({"A": 1 / 2, "B": 2 / 2}, {"A": 0.9, "B": 0.3}),
        "barbaz": ({"A": 3 / 3, "B": 1 / 3}, {"A": 0.3, "B": 0.2}),
        "foofoo": ({"A": 1 / 1, "B": 0 / 1}, {"A": 0.6, "B": 0.1}),
        "barbar": ({"A": 0 / 1, "B": 1 / 1}, {"A": 0.2, "B": 0.9}),
    }

    class FakeTextClassifier(FakePredictMixinABC, TextClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            fake_predict_one_dp(dp, label_name)

    class FakeAETClassifier(FakePredictMixinABC, AETClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            fake_predict_one_dp(dp, label_name)

    class FakeLATClassifier(FakePredictMixinABC, LATClassifier):
        def fake_predict_one_dp(self, dp, label_name):
            fake_predict_one_dp(dp, label_name)

    def fake_predict_one_dp(dp, label_name):
        dp.add_label(
            label_name, "A", input2label2p[dp.get_metadata("orig_text")][1]["A"]
        )
        dp.add_label(
            label_name, "B", input2label2p[dp.get_metadata("orig_text")][1]["B"]
        )

    mocker.patch(
        f"run_training.{corpus_cls_name}.make_classifier",
        build_fake_make_classifier(
            corpus_cls_name,
            FakeAETClassifier,
            FakeLATClassifier,
            FakeTextClassifier,
            anns=("1", "2"),
        ),
    )
    artifacts_dir = tmp_path / "artifacts"

    train_one_epoch(
        tmp_path,
        artifacts_dir,
        method,
        use_lora=use_lora,
        lora_target=lora_target,
    )
    label_dict, inputs, hlv = load_artifacts(artifacts_dir, eval_split)
    true_hlv, pred_hlv = hlv

    assert len(label_dict) == 2
    assert label_dict.get_idx_for_items(("A", "B")) == [0, 1]
    for text, (true_label2p, pred_label2p) in input2label2p.items():
        idx = inputs.index({"text": text})
        assert true_hlv[idx] == pytest.approx([true_label2p["A"], true_label2p["B"]])
        assert pred_hlv[idx] == pytest.approx([pred_label2p["A"], pred_label2p["B"]])


@pytest.mark.parametrize(
    "method,corpus_cls_name",
    list(
        zip(
            "ReL MV AL SL SLMV AE AEh SmF1 SMF1 JSD LA-min LA-max LA-mean".split(),
            [
                "DCorpus",
                "MVCorpus",
                "ALCorpus",
                "SLCorpus",
                "SLMVCorpus",
                "AnnLCorpus",
                "AnnLCorpus",
                "SLCorpus",
                "SLCorpus",
                "SLCorpus",
                "ACCorpus",
                "ACCorpus",
                "ACCorpus",
            ],
        )
    ),
)
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_nli_various_methods_work(
    tmp_path, write_nli_jsonl, mocker, method, corpus_cls_name, use_lora, lora_target
):
    write_nli_jsonl(["foo bar E 1"], "train.jsonl")
    mock_corpus_cls = mocker.patch(f"run_training.{corpus_cls_name}", autospec=True)
    mock_trainer = mocker.patch("run_training.ModelTrainer", autospec=True)

    train_one_epoch(
        tmp_path,
        tmp_path / "artifacts",
        method,
        use_lora=use_lora,
        lora_target=lora_target,
    )

    mock_trainer.assert_called_once_with(
        mock_corpus_cls.return_value.make_classifier.return_value,
        mock_corpus_cls.return_value,
    )


@pytest.mark.parametrize(
    "method,corpus_cls_name",
    list(
        zip(
            "ReL MV AL SL SLMV AE AEh AR ARh SmF1 SMF1 JSD LA-min LA-max LA-mean".split(),
            [
                "DCorpus",
                "MVCorpus",
                "ALCorpus",
                "SLCorpus",
                "SLMVCorpus",
                "AnnLCorpus",
                "AnnLCorpus",
                "ARCorpus",
                "ARCorpus",
                "SLCorpus",
                "SLCorpus",
                "SLCorpus",
                "ACCorpus",
                "ACCorpus",
                "ACCorpus",
            ],
        )
    ),
)
def test_binary_classification_various_methods_work(
    tmp_path, write_bc_jsonl, mocker, method, corpus_cls_name
):
    write_bc_jsonl(["foobar 1 A"], "train.jsonl")
    mock_corpus_cls = mocker.patch(f"run_training.{corpus_cls_name}", autospec=True)
    mock_trainer = mocker.patch("run_training.ModelTrainer", autospec=True)

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)

    mock_trainer.assert_called_once_with(
        mock_corpus_cls.return_value.make_classifier.return_value,
        mock_corpus_cls.return_value,
    )


@pytest.mark.parametrize(
    "method,corpus_cls_name",
    list(
        zip(
            "ReL MV SL SLMV AE AEh AR ARh SmF1 SmF1-v2 SMF1 JSD LA-min LA-max LA-mean".split(),
            "DCorpus MVCorpus SLCorpus SLMVCorpus AnnLCorpus AnnLCorpus ARCorpus ARCorpus SLCorpus SLCorpus SLCorpus SLCorpus ACCorpus ACCorpus ACCorpus".split(),
        )
    ),
)
@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("lora_target", [None, "all-linear"])
def test_multilabel_classification_various_methods_work(
    tmp_path,
    write_mullab_jsonl,
    mocker,
    method,
    corpus_cls_name,
    use_lora,
    lora_target,
):
    write_mullab_jsonl(["foobar B;1"], "train.jsonl")
    mock_corpus_cls = mocker.patch(f"run_training.{corpus_cls_name}", autospec=True)
    mock_trainer = mocker.patch("run_training.ModelTrainer", autospec=True)

    train_one_epoch(
        tmp_path,
        tmp_path / "artifacts",
        method,
        use_lora=use_lora,
        lora_target=lora_target,
    )

    clf = mock_corpus_cls.return_value.make_classifier.return_value
    mock_trainer.assert_called_once_with(clf, mock_corpus_cls.return_value)
    if method == "LA-max":
        assert clf.loss_function.aggfunc == "max"
    elif method == "LA-mean":
        assert clf.loss_function.aggfunc == "mean"


@pytest.mark.parametrize(
    "method", "ReL MV AL SL SLMV AE AEh SmF1 SMF1 JSD LA-min LA-max LA-mean".split()
)
def test_nli_trailing_whitespace_in_test_inputs_preserved(
    write_nli_jsonl, tmp_path, mocker, method
):
    write_nli_jsonl(["foo bar N 1", "bar quux E 2"], "train.jsonl")
    with (tmp_path / "test.jsonl").open("w", encoding="utf8") as f:
        print(
            json.dumps(
                {
                    "input": {"premise": "foo ", "hypothesis": "bar "},
                    "output": {"verdict": "N"},
                }
            ),
            file=f,
        )
    mocker.patch("run_training.ModelTrainer", autospec=True)

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)

    assert json.loads((tmp_path / "artifacts" / "inputs.jsonl").read_text("utf8")) == {
        "premise": "foo ",
        "hypothesis": "bar ",
    }


@pytest.mark.parametrize(
    "method",
    "ReL MV AL SL SLMV AE AEh AR ARh SmF1 SMF1 JSD LA-min LA-max LA-mean".split(),
)
def test_binary_classification_trailing_whitespace_in_test_inputs_preserved(
    write_bc_jsonl, tmp_path, mocker, method
):
    write_bc_jsonl(["foobar 0 A", "barquux 1 B"], "train.jsonl")
    with (tmp_path / "test.jsonl").open("w", encoding="utf8") as f:
        print(
            json.dumps(
                {
                    "input": {"text": "foobar "},
                    "output": {"class": False},
                }
            ),
            file=f,
        )
    mocker.patch("run_training.ModelTrainer", autospec=True)

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)

    assert json.loads((tmp_path / "artifacts" / "inputs.jsonl").read_text("utf8")) == {
        "text": "foobar "
    }


@pytest.mark.parametrize(
    "method",
    "ReL MV SL SLMV AE AEh AR ARh SmF1 SmF1-v2 SMF1 JSD LA-min LA-max LA-mean".split(),
)
def test_multilabel_classification_trailing_whitespace_in_test_inputs_preserved(
    write_mullab_jsonl, tmp_path, mocker, method
):
    write_mullab_jsonl(["foobar A;1", "barquux B;2"], "train.jsonl")
    with (tmp_path / "test.jsonl").open("w", encoding="utf8") as f:
        print(
            json.dumps(
                {
                    "input": {"text": "foobar "},
                    "output": {"classes": ["B"]},
                }
            ),
            file=f,
        )
    mocker.patch("run_training.ModelTrainer", autospec=True)

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)

    assert json.loads((tmp_path / "artifacts" / "inputs.jsonl").read_text("utf8")) == {
        "text": "foobar "
    }


@pytest.mark.slow
@pytest.mark.parametrize("method", ["AR", "ARh"])
def test_binary_classification_annotator_ranking_method_with_dev_set(
    write_bc_jsonl, tmp_path, method
):
    write_bc_jsonl(["foobar 0 A", "barbar 1 B"], "train.jsonl")
    write_bc_jsonl(["foobar 0 A", "barbar 1 B"], "dev.jsonl")

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["AR", "ARh"])
def test_multilabel_classification_annotator_ranking_method_with_dev_set(
    write_mullab_jsonl, tmp_path, method
):
    write_mullab_jsonl(["foobar A;1", "barbar B;2"], "train.jsonl")
    write_mullab_jsonl(["foobar A;1", "barbar B;2"], "dev.jsonl")

    train_one_epoch(tmp_path, tmp_path / "artifacts", method)


@pytest.mark.slow
@pytest.mark.parametrize(
    "method", "ReL MV AL SL SLMV AE AEh SmF1 SMF1 JSD LA-min LA-max LA-mean".split()
)
def test_nli_delete_model_file_when_finish(write_nli_jsonl, tmp_path, method):
    write_nli_jsonl(["foo bar N 1"], "train.jsonl")

    train_one_epoch(tmp_path, tmp_path / "artifacts", save_model=False)

    assert not (tmp_path / "artifacts" / "final-model.pt").exists()


@pytest.mark.slow
@pytest.mark.parametrize(
    "method",
    "ReL MV AL SL SLMV AE AEh AR ARh SmF1 SMF1 JSD LA-min LA-max LA-mean".split(),
)
def test_binary_classification_delete_model_file_when_finish(
    write_bc_jsonl, tmp_path, method
):
    write_bc_jsonl(["foobar 0 A"], "train.jsonl")

    train_one_epoch(tmp_path, tmp_path / "artifacts", save_model=False)

    assert not (tmp_path / "artifacts" / "final-model.pt").exists()


@pytest.mark.slow
@pytest.mark.parametrize(
    "method",
    "ReL MV SL SLMV AE AEh AR ARh SmF1 SmF1-v2 SMF1 JSD LA-min LA-max LA-mean".split(),
)
def test_multilabel_classification_delete_model_file_when_finish(
    write_mullab_jsonl, tmp_path, method
):
    write_mullab_jsonl(["foobar A;1"], "train.jsonl")

    train_one_epoch(tmp_path, tmp_path / "artifacts", save_model=False)

    assert not (tmp_path / "artifacts" / "final-model.pt").exists()


@pytest.mark.slow
@pytest.mark.parametrize(
    "method", "ReL MV SL SLMV AE AEh AR ARh SmF1 SMF1 JSD LA-min LA-max LA-mean".split()
)
def test_multilabel_classification_eval_on_epoch_finished(
    write_mullab_jsonl, tmp_path, mocker, method
):
    write_mullab_jsonl(["foo A;1", "baz A B;2"], "train.jsonl")
    write_mullab_jsonl(["foo;", "foo B;", "foo A B;", "bar A;"], "dev.jsonl")
    write_mullab_jsonl(["foo A;", "foo;", "foo A B;", "bar A;"], "test.jsonl")
    mock_run = mocker.Mock(spec=Run)

    train(
        tmp_path,
        tmp_path / "artifacts",
        method,
        hide_eval_sets_from_trainer=True,
        labels=("A", "B"),
        max_epochs=2,
        _run=mock_run,
        eval_on_epoch_finished=True,
    )

    metric_name2call_args_list = {
        name: [
            (args, kwargs)
            for (args, kwargs) in mock_run.log_scalar.call_args_list
            if args[0].startswith("dev/") and args[0][len("dev/") :] == name
        ]
        for name in "ent_corr hard_micro_f1 poJSD soft_micro_f1".split()
    }
    for call_args_list in metric_name2call_args_list.values():
        assert len(call_args_list) == 2


def train(data_dir, artifacts_dir, *args, **kwargs):
    return run_train(str(data_dir), str(artifacts_dir), *args, **kwargs)


def train_one_epoch(*args, **kwargs):
    kwargs["max_epochs"] = 1
    return train(*args, **kwargs)


class FakePredictMixinABC(abc.ABC):
    # Mypy complains if 'fake_predict' is renamed to 'predict'
    def predict(self, *args, **kwargs):
        return self.fake_predict(*args, **kwargs)

    def fake_predict(
        self, datapoints, *args, label_name="", return_loss=False, **kwargs
    ):
        for dp in datapoints:
            self.fake_predict_one_dp(dp, label_name)
        if return_loss:  # Avoid Flair error when dev set exists
            return 0.0

    @abc.abstractmethod
    def fake_predict_one_dp(self, dp, label_name):
        raise NotImplementedError


def build_fake_make_classifier(
    corpus_cls_name,
    fake_ae_classifier_cls,
    fake_la_classifier_cls,
    fake_classifier_cls,
    anns,
):
    def fake_make_classifier(self, emb, labdict):
        if corpus_cls_name == "AnnLCorpus":
            ann_dict = Dictionary()
            for ann in anns:
                ann_dict.add_item(ann)
            return fake_ae_classifier_cls(
                emb,
                self.hard_label_type,
                labdict,
                f"{self.hard_label_type} by annotator {{0}}",
                ann_dict,
            )
        elif corpus_cls_name == "ACCorpus":
            return fake_la_classifier_cls(
                emb, label_type="@@UNUSED@@", label_dictionary=labdict
            )
        else:
            return fake_classifier_cls(
                emb,
                label_type=self.soft_label_type
                if corpus_cls_name == "SLCorpus"  # only soft label type exists
                else self.hard_label_type,
                label_dictionary=labdict,
            )

    return fake_make_classifier


def load_artifacts(artifacts_dir, eval_split):
    if eval_split == "test":
        inputs_filename, hlv_filename = "inputs.jsonl", "hlv.npy"
    elif eval_split == "dev":
        inputs_filename, hlv_filename = "dev-inputs.jsonl", "dev-hlv.npy"
    else:
        raise ValueError(f"unrecognised eval split: {eval_split}")
    label_dict = Dictionary.load_from_file(artifacts_dir / "label.dict")
    inputs = [
        json.loads(l)
        for l in (artifacts_dir / inputs_filename)
        .read_text("utf8")
        .strip()
        .splitlines()
    ]
    hlv = np.load(artifacts_dir / hlv_filename)
    return label_dict, inputs, hlv
