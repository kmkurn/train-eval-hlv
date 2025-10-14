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

import contextlib

import numpy as np
import pytest
from flair.data import Dictionary
from run_eval import evaluate
from sacred.run import Run


@pytest.fixture
def mock_eval_funcs(mocker):
    @contextlib.contextmanager
    def _mock_eval_funcs(overall_metric_names, true_hlv, pred_hlv, label_dict):
        overall_metric_rvs = np.random.rand(len(overall_metric_names))
        mocks = [
            mocker.patch(f"run_eval.{name}", autospec=True, return_value=rv)
            for name, rv in zip(overall_metric_names, overall_metric_rvs)
        ]
        mock_hard_prfs = mocker.patch(
            "run_eval.hard_prfs", return_value=np.random.rand(4, len(label_dict))
        )
        mock_soft_prfs = mocker.patch(
            "run_eval.soft_prfs", return_value=np.random.rand(4, len(label_dict))
        )
        mock_run = mocker.Mock(spec=Run)
        mocks.extend([mock_hard_prfs, mock_soft_prfs])

        yield mock_run

        for name, mock in zip(overall_metric_names, mocks):
            assert mock.call_count == 1
            assert mock.call_args == mocker.call(
                pytest.approx(true_hlv), pytest.approx(pred_hlv)
            )
            mock_run.log_scalar.call_args_list.remove(
                mocker.call(name, pytest.approx(mock.return_value))
            )
        for t, mock in zip("hard soft".split(), [mock_hard_prfs, mock_soft_prfs]):
            for i, x in enumerate("p r f1 supp".split()):
                if x != "supp":
                    mock_run.log_scalar.call_args_list.remove(
                        mocker.call(
                            f"macro/{t}_{x}", pytest.approx(mock.return_value[i].mean())
                        )
                    )
                for j in range(len(label_dict)):
                    l = label_dict.get_item_for_index(j)
                    mock_run.log_scalar.call_args_list.remove(
                        mocker.call(
                            f"{t}_{x}/{l}", pytest.approx(mock.return_value[i, j])
                        )
                    )
        assert not mock_run.log_scalar.call_args_list

    return _mock_eval_funcs


@pytest.mark.parametrize("true_hlv_saved", [False, True])
def test_nli_ok(write_nli_jsonl, tmp_path, mock_eval_funcs, mocker, true_hlv_saved):
    write_nli_jsonl(
        ["foo bar E", "foo bar N", "foo bar N", "bar baz E", "bar baz N"], "test.jsonl"
    )
    write_nli_jsonl(["bar baz", "foo bar"], "inputs.jsonl")
    d = Dictionary(add_unk=False)
    d.add_item("N")
    d.add_item("E")
    d.save(tmp_path / "label.dict")
    pred_hlv = np.array([[0.3, 0.7], [0.6, 0.4]])
    true_hlv = np.array([[1 / 2, 1 / 2], [2 / 3, 1 / 3]])
    if true_hlv_saved:
        hlv = np.stack((true_hlv, pred_hlv), axis=0)
    else:
        hlv = pred_hlv
    np.save(tmp_path / "hlv.npy", hlv)
    with mock_eval_funcs(
        "hard_acc poJSD ent_corr soft_acc".split(), true_hlv, pred_hlv, d
    ) as mock_run:
        if true_hlv_saved:
            evaluate(str(tmp_path), _run=mock_run)
        else:
            evaluate(
                str(tmp_path),
                str(tmp_path / "test.jsonl"),
                _run=mock_run,
            )


@pytest.mark.parametrize("true_hlv_saved", [False, True])
def test_bc_ok(write_bc_jsonl, tmp_path, mock_eval_funcs, mocker, true_hlv_saved):
    write_bc_jsonl(
        ["foobar 1", "foobar 0", "foobar 0", "barbaz 1", "barbaz 0"], "test.jsonl"
    )
    write_bc_jsonl(["barbaz", "foobar"], "inputs.jsonl")
    d = Dictionary(add_unk=False)
    d.add_item("False")
    d.add_item("True")
    d.save(tmp_path / "label.dict")
    pred_hlv = np.array([[0.3, 0.7], [0.6, 0.4]])
    true_hlv = np.array([[1 / 2, 1 / 2], [2 / 3, 1 / 3]])
    if true_hlv_saved:
        hlv = np.stack((true_hlv, pred_hlv), axis=0)
    else:
        hlv = pred_hlv
    np.save(tmp_path / "hlv.npy", hlv)
    with mock_eval_funcs(
        "hard_acc poJSD ent_corr soft_acc".split(), true_hlv, pred_hlv, d
    ) as mock_run:
        if true_hlv_saved:
            evaluate(str(tmp_path), _run=mock_run)
        else:
            evaluate(
                str(tmp_path),
                str(tmp_path / "test.jsonl"),
                _run=mock_run,
            )


def test_mullab_ok(write_mullab_jsonl, tmp_path, mock_eval_funcs, mocker):
    write_mullab_jsonl(["foo", "bar"], "inputs.jsonl")
    d = Dictionary(add_unk=False)
    d.add_item("A")
    d.add_item("B")
    d.save(tmp_path / "label.dict")
    pred_hlv = np.array([[0.3, 0.9], [0.6, 0.7]])
    true_hlv = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    hlv = np.stack((true_hlv, pred_hlv), axis=0)
    np.save(tmp_path / "hlv.npy", hlv)
    with mock_eval_funcs(
        "hard_micro_f1 ent_corr soft_micro_f1".split(), true_hlv, pred_hlv, d
    ) as mock_run:
        mock_pojsd = mocker.patch(
            "run_eval.poJSD", autospec=True, return_value=0.27856
        )
        evaluate(str(tmp_path), multilabel=True, _run=mock_run)
        assert mock_pojsd.call_args_list == [
            mocker.call(
                pytest.approx(true_hlv), pytest.approx(pred_hlv), multilabel=True
            ),
        ]
        mock_run.log_scalar.call_args_list.remove(
            mocker.call("poJSD", pytest.approx(0.27856))
        )
