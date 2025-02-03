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

import pickle

import numpy as np
from compute_method2hlv import main


def test_correct(tmp_path):
    true_hlv = np.array(
        [
            [0.6, 0.2],
            [0.7, 0.9],
        ]
    )
    pred_hlv = {"a": {}, "b": {}}
    pred_hlv["a"][1] = np.array(
        [
            [0.3, 0.3],
            [0.9, 0.8],
        ]
    )
    pred_hlv["a"][2] = np.array(
        [
            [0.5, 0.6],
            [0.8, 0.1],
        ]
    )
    pred_hlv["b"][1] = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.2],
        ]
    )
    pred_hlv["b"][2] = np.array(
        [
            [0.4, 0.9],
            [0.3, 0.5],
        ]
    )
    for meth in ("a", "b"):
        for run_id in (1, 2):
            (tmp_path / meth / f"run{run_id}").mkdir(parents=True)
            np.save(
                tmp_path / meth / f"run{run_id}" / "hlv.npy",
                np.stack([true_hlv, pred_hlv[meth][run_id]]),
            )

    main(
        tmp_path,
        methods=("a", "b"),
        n_runs=2,
        output_file=tmp_path / "method2hlv.pkl",
    )

    with (tmp_path / "method2hlv.pkl").open("rb") as f:
        method2hlv = pickle.load(f)
    assert np.allclose(
        method2hlv["a"], np.stack([true_hlv, np.array([[0.4, 0.45], [0.85, 0.45]])])
    )
    assert np.allclose(
        method2hlv["b"], np.stack([true_hlv, np.array([[0.25, 0.85], [0.25, 0.35]])])
    )
