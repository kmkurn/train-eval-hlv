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
import csv
import json
import pickle
import random
from itertools import combinations
from pathlib import Path
from typing import Optional

from flair.data import Dictionary


def main(
    method2hlv_file: Path,
    inputs_file: Path,
    label_dict_file: Path,
    output_dir: Path,
    thresh: float,
    size: Optional[int] = None,
    shuffle: bool = False,
    randomise_options: bool = False,
    start_id: int = 0,
    shuffle_within_method: bool = False,
    vs: str = "true",
) -> None:
    with method2hlv_file.open("rb") as f:
        method2hlv = pickle.load(f)
    with inputs_file.open(encoding="utf8") as f:
        inputs = [json.loads(l) for l in f]
    if size is None:
        size = len(inputs)
    label_dict = Dictionary.load_from_file(label_dict_file)

    output_dir.mkdir()
    count = 0
    data, metadata = [], []
    indices = list(reversed(range(len(inputs))))
    if shuffle_within_method:
        random.shuffle(indices)
    if vs == "true":
        for meth, (true_hlv, pred_hlv) in method2hlv.items():
            selected_indices: list[int] = []
            while len(selected_indices) < size:
                try:
                    idx = indices.pop()
                except IndexError:
                    indices = list(reversed(range(len(inputs))))
                    if shuffle_within_method:
                        random.shuffle(indices)
                else:
                    selected_indices.append(idx)
            for i in selected_indices:
                swapped = random.choice([False, True]) if randomise_options else False
                hlvA, hlvB = (pred_hlv, true_hlv) if swapped else (true_hlv, pred_hlv)
                data.append(
                    {
                        "id": start_id + count,
                        "text": inputs[i]["text"],
                        "optionA": sorted(
                            [
                                {"areaOfLaw": l, "confidence": c}
                                for l in label_dict.get_items()
                                if (c := hlvA[i, label_dict.get_idx_for_item(l)])
                                > thresh
                            ],
                            key=lambda x: x["confidence"],
                            reverse=True,
                        ),
                        "optionB": sorted(
                            [
                                {"areaOfLaw": l, "confidence": c}
                                for l in label_dict.get_items()
                                if (c := hlvB[i, label_dict.get_idx_for_item(l)])
                                > thresh
                            ],
                            key=lambda x: x["confidence"],
                            reverse=True,
                        ),
                    }
                )
                metadata.append(
                    {
                        "id": start_id + count,
                        "method": meth,
                        "trueHLV": "B" if swapped else "A",
                    }
                )
                count += 1
    else:
        assert vs == "method"
        for methA, methB in combinations(method2hlv.keys(), 2):
            selected_indices = []
            while len(selected_indices) < size:
                try:
                    idx = indices.pop()
                except IndexError:
                    indices = list(reversed(range(len(inputs))))
                    if shuffle_within_method:
                        random.shuffle(indices)
                else:
                    selected_indices.append(idx)
            for i in selected_indices:
                swapped = random.choice([False, True]) if randomise_options else False
                methA, methB = (methB, methA) if swapped else (methA, methB)
                hlvA, hlvB = method2hlv[methA][1], method2hlv[methB][1]
                data.append(
                    {
                        "id": start_id + count,
                        "text": inputs[i]["text"],
                        "optionA": sorted(
                            [
                                {"areaOfLaw": l, "confidence": c}
                                for l in label_dict.get_items()
                                if (c := hlvA[i, label_dict.get_idx_for_item(l)])
                                > thresh
                            ],
                            key=lambda x: x["confidence"],
                            reverse=True,
                        ),
                        "optionB": sorted(
                            [
                                {"areaOfLaw": l, "confidence": c}
                                for l in label_dict.get_items()
                                if (c := hlvB[i, label_dict.get_idx_for_item(l)])
                                > thresh
                            ],
                            key=lambda x: x["confidence"],
                            reverse=True,
                        ),
                    }
                )
                metadata.append(
                    {"id": start_id + count, "optionA": methA, "optionB": methB}
                )
                count += 1
    if shuffle:
        random.shuffle(data)
    with (output_dir / "input.jsonl").open("w", encoding="utf8") as f:
        for dat in data:
            print(json.dumps(dat), file=f)
    with (output_dir / "metadata.csv").open("w", encoding="utf8") as f:
        writer = csv.DictWriter(
            f, ("id method trueHLV" if vs == "true" else "id optionA optionB").split()
        )
        writer.writeheader()
        writer.writerows(metadata)
    with (output_dir / "config.txt").open("w", encoding="utf8") as f:
        print(f"{thresh=}", file=f)
        print(f"{size=}", file=f)
        print(f"{shuffle=}", file=f)
        print(f"{randomise_options=}", file=f)
        print(f"{shuffle_within_method=}", file=f)
        print(f"{vs=}", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create HLV annotation input data")
    parser.add_argument("method2hlv", type=Path, help="method-to-HLV mapping (.pkl)")
    parser.add_argument(
        "inputs",
        type=Path,
        help="JSONL file whose lines correspond to rows in HLV in method2hlv",
    )
    parser.add_argument(
        "label_dict",
        type=Path,
        help="Flair's Dictionary of labels corresponding to columns in HLV in method2hlv",
    )
    parser.add_argument("output_dir", type=Path, help="save outputs to this directory")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.0,
        help="omit labels scoring less than this number",
    )
    parser.add_argument("--size", type=int, help="number of examples for each method")
    parser.add_argument(
        "--shuffle", action="store_true", help="shuffle the output rows"
    )
    parser.add_argument(
        "--randomise-options", action="store_true", help="randomise options A and B"
    )
    parser.add_argument(
        "--start-id", type=int, default=0, help="start ID from this number"
    )
    parser.add_argument(
        "--shuffle-within-method",
        action="store_true",
        help="shuffle examples within a method",
    )
    parser.add_argument(
        "--vs",
        default="true",
        choices=["true", "method"],
        help="comparing against what?",
    )
    args = parser.parse_args()
    main(
        args.method2hlv,
        args.inputs,
        args.label_dict,
        args.output_dir,
        args.threshold,
        args.size,
        args.shuffle,
        args.randomise_options,
        args.start_id,
        args.shuffle_within_method,
        args.vs,
    )
