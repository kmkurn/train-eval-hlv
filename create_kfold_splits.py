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
import json
import random
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


def main(
    file_: Path,
    output_dir: Path,
    fold: int,
    shuffle: bool = False,
    is_nli: bool = True,
    stratified: bool = False,
) -> None:
    with file_.open(encoding="utf8") as f:
        data = list(map(json.loads, f))

    if stratified:
        if is_nli:
            raise NotImplementedError
        if shuffle:
            random.shuffle(data)
        sgkf = StratifiedGroupKFold(n_splits=fold, shuffle=shuffle)
        hashable_labels = [" ".join(sorted(obj["output"]["classes"])) for obj in data]
        hashable_groups = [obj["input"]["text"] for obj in data]
        for i, (train_ixs, test_ixs) in enumerate(
            sgkf.split(data, hashable_labels, hashable_groups)
        ):
            (output_dir / str(i)).mkdir()
            for ixs, set_ in zip([train_ixs, test_ixs], ["train", "test"]):
                with (output_dir / str(i) / f"{set_}.jsonl").open(
                    "w", encoding="utf8"
                ) as f:
                    for ix in ixs:
                        print(json.dumps(data[ix]), file=f)
    else:
        get_hashable_input = _get_textpair if is_nli else _get_text
        inputs = list(set(map(get_hashable_input, data)))
        if shuffle:
            random.shuffle(inputs)
        fold_lens = []
        for i in range(len(inputs) % fold):
            fold_lens.append(len(inputs) // fold + 1)
        while len(fold_lens) < fold:
            fold_lens.append(len(inputs) // fold)

        offset = 0
        Indices = list[int]
        train_ix_ls: list[Indices] = []
        test_ix_ls: list[Indices] = []
        for len_ in fold_lens:
            test_inputs = inputs[offset : offset + len_]
            train_inputs = inputs[:offset]
            train_inputs.extend(inputs[offset + len_ :])
            test_ix_ls.append(
                [
                    i
                    for i, obj in enumerate(data)
                    if get_hashable_input(obj) in set(test_inputs)
                ]
            )
            train_ix_ls.append(
                [
                    i
                    for i, obj in enumerate(data)
                    if get_hashable_input(obj) in set(train_inputs)
                ]
            )
            offset += len_

        for i in range(fold):
            (output_dir / str(i)).mkdir()
            for ixs, set_ in zip([train_ix_ls, test_ix_ls], ["train", "test"]):
                with (output_dir / str(i) / f"{set_}.jsonl").open(
                    "w", encoding="utf8"
                ) as f:
                    for ix in ixs[i]:
                        print(json.dumps(data[ix]), file=f)


def _get_textpair(obj: dict[str, dict]) -> tuple[str, str]:
    return obj["input"]["premise"], obj["input"]["hypothesis"]


def _get_text(obj: dict[str, dict]) -> str:
    return obj["input"]["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create K-fold CV splits.")
    parser.add_argument("file", type=Path, help="dataset file in our JSONL format")
    parser.add_argument("output_dir", type=Path, help="output directory")
    parser.add_argument("fold", type=int, help="number of folds (>1)")
    parser.add_argument(
        "--shuffle", action="store_true", help="shuffle before splitting"
    )
    parser.add_argument(
        "--nli", action="store_true", help="flag for NLI task"
    )
    parser.add_argument(
        "--stratified", action="store_true", help="perform stratified K-fold"
    )
    args = parser.parse_args()
    if args.fold <= 1:
        sys.exit(f"Error: fold must be >1, found {args.fold}")
    main(args.file, args.output_dir, args.fold, args.shuffle, args.nli, args.stratified)
