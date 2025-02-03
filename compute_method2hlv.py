#!/usr/bin/env python

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

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np


def main(
    root_dir: Path, methods: Iterable[str], n_runs: int, output_file: Path
) -> None:
    with output_file.open("wb") as f:
        pickle.dump(
            {
                meth: np.stack(
                    [
                        np.load(root_dir / meth / f"run{id_}" / "hlv.npy")
                        for id_ in range(1, n_runs + 1)
                    ],
                    axis=0,
                ).mean(axis=0)
                for meth in methods
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute method-to-mean-HLV mapping")
    parser.add_argument(
        "-r",
        "--root-dir",
        required=True,
        type=Path,
        help="where <method>/run<run_id>/hlv.npy lives",
    )
    parser.add_argument(
        "-m",
        "--methods",
        nargs="+",
        default="ReL AL MV SL SLMV AE AEh AR ARh".split(),
    )
    parser.add_argument("-n", "--n-runs", type=int, default=3, help="number of runs")
    parser.add_argument(
        "-o", "--output-file", required=True, type=Path, help="output file"
    )
    args = parser.parse_args()
    main(args.root_dir, args.methods, args.n_runs, args.output_file)
