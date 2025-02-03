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

import numpy as np
import pytest
from flair.data import Dictionary
from run_eval import evaluate
from sacred.run import Run


@pytest.mark.parametrize("true_hlv_saved", [False, True])
def test_nli_ok(write_nli_jsonl, tmp_path, mocker, true_hlv_saved):
    write_nli_jsonl(
        ["foo bar E", "foo bar N", "foo bar N", "bar baz E", "bar baz N"], "test.jsonl"
    )
    write_nli_jsonl(["bar baz", "foo bar"], "inputs.jsonl")
    d = Dictionary(add_unk=False)
    d.add_item("N")
    d.add_item("E")
    d.save(tmp_path / "label.dict")
    pred_hlv = np.array([[0.3, 0.7], [0.6, 0.4]])
    if true_hlv_saved:
        true_hlv = np.array([[1 / 2, 1 / 2], [2 / 3, 1 / 3]])
        hlv = np.stack((true_hlv, pred_hlv), axis=0)
    else:
        hlv = pred_hlv
    np.save(tmp_path / "hlv.npy", hlv)
    name2mock = {
        name: mocker.patch(f"run_eval.{name}", autospec=True, return_value=rv)
        for name, rv in zip(
            "hard_acc poJSD ent_corr soft_acc".split(),
            (
                0.357878,
                0.27856,
                -0.458628,
                0.736556,
                np.random.rand(4, 2),
                np.random.rand(4, 2),
            ),
        )
    }
    mock_hard_prfs = mocker.patch(
        "run_eval.hard_prfs", return_value=np.random.rand(4, 2)
    )
    mock_soft_prfs = mocker.patch(
        "run_eval.soft_prfs", return_value=np.random.rand(4, 2)
    )
    mock_run = mocker.Mock(spec=Run)

    if true_hlv_saved:
        evaluate(str(tmp_path), _run=mock_run)
    else:
        evaluate(
            str(tmp_path),
            str(tmp_path / "test.jsonl"),
            _run=mock_run,
        )

    for mock in list(name2mock.values()) + [mock_hard_prfs, mock_soft_prfs]:
        assert mock.call_count == 1
        assert len(mock.call_args.args) == 2
        assert mock.call_args.args[0] == pytest.approx(
            np.array([[1 / 2, 1 / 2], [2 / 3, 1 / 3]])
        )
        assert mock.call_args.args[1] == pytest.approx(
            np.array([[0.3, 0.7], [0.6, 0.4]])
        )
    for name, mock in name2mock.items():
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
            for j, l in enumerate(("N", "E")):
                mock_run.log_scalar.call_args_list.remove(
                    mocker.call(f"{t}_{x}/{l}", pytest.approx(mock.return_value[i, j]))
                )
    assert not mock_run.log_scalar.call_args_list


@pytest.mark.parametrize("true_hlv_saved", [False, True])
def test_bc_ok(write_bc_jsonl, tmp_path, mocker, true_hlv_saved):
    write_bc_jsonl(
        ["foobar 1", "foobar 0", "foobar 0", "barbaz 1", "barbaz 0"], "test.jsonl"
    )
    write_bc_jsonl(["barbaz", "foobar"], "inputs.jsonl")
    d = Dictionary(add_unk=False)
    d.add_item("False")
    d.add_item("True")
    d.save(tmp_path / "label.dict")
    pred_hlv = np.array([[0.3, 0.7], [0.6, 0.4]])
    if true_hlv_saved:
        true_hlv = np.array([[1 / 2, 1 / 2], [2 / 3, 1 / 3]])
        hlv = np.stack((true_hlv, pred_hlv), axis=0)
    else:
        hlv = pred_hlv
    np.save(tmp_path / "hlv.npy", hlv)
    name2mock = {
        name: mocker.patch(f"run_eval.{name}", autospec=True, return_value=rv)
        for name, rv in zip(
            "hard_acc poJSD ent_corr soft_acc".split(),
            (
                0.357878,
                0.27856,
                -0.458628,
                0.736556,
                np.random.rand(4, 2),
                np.random.rand(4, 2),
            ),
        )
    }
    mock_hard_prfs = mocker.patch(
        "run_eval.hard_prfs", return_value=np.random.rand(4, 2)
    )
    mock_soft_prfs = mocker.patch(
        "run_eval.soft_prfs", return_value=np.random.rand(4, 2)
    )
    mock_run = mocker.Mock(spec=Run)

    if true_hlv_saved:
        evaluate(str(tmp_path), _run=mock_run)
    else:
        evaluate(
            str(tmp_path),
            str(tmp_path / "test.jsonl"),
            _run=mock_run,
        )

    for mock in list(name2mock.values()) + [mock_hard_prfs, mock_soft_prfs]:
        assert mock.call_count == 1
        assert len(mock.call_args.args) == 2
        assert mock.call_args.args[0] == pytest.approx(
            np.array([[1 / 2, 1 / 2], [2 / 3, 1 / 3]])
        )
        assert mock.call_args.args[1] == pytest.approx(
            np.array([[0.3, 0.7], [0.6, 0.4]])
        )
    for name, mock in name2mock.items():
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
            for j, l in enumerate(("False", "True")):
                mock_run.log_scalar.call_args_list.remove(
                    mocker.call(f"{t}_{x}/{l}", pytest.approx(mock.return_value[i, j]))
                )
    assert not mock_run.log_scalar.call_args_list
