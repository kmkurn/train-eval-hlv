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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from flair.data import Dictionary


def main(
    input_dir: Path,
    output_dir: Path,
    min_llama_input_id: int,
    data_filename: str = "data.jsonl",
) -> None:
    data = defaultdict(list)
    with (input_dir / data_filename).open(encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            data["input_id"].append(obj["inputId"])
            data["annotator_id"].append(obj["annotatorId"])
            data["timestamp"].append(pd.to_datetime(obj["timestamp"]))
            data["hlv_annotation"].append(
                (
                    obj["preferredHLVAnnotation"]["isOptionAPreferred"],
                    obj["preferredHLVAnnotation"]["isOptionBPreferred"],
                )
            )
    df_data = pd.DataFrame(data)
    id2annots = (
        df_data.sort_values("timestamp", ascending=False)
        .drop_duplicates(["input_id", "annotator_id"])
        .groupby("input_id")["hlv_annotation"]
        .agg(list)
        .to_dict()
    )

    metadata = pd.read_csv(input_dir / "metadata.csv", index_col="id")
    metadata["model"] = metadata.index.map(
        lambda i: "roberta" if i < min_llama_input_id else "llama"
    )

    method_dict = Dictionary(add_unk=False)
    for m in metadata.optionA.unique():
        method_dict.add_item(m)
    for m in metadata.optionB.unique():
        method_dict.add_item(m)

    ROBERTA, LLAMA = 0, 1
    comps = np.zeros([2, len(method_dict), len(method_dict)], dtype=int)
    wins = np.zeros([2, len(method_dict), len(method_dict)], dtype=int)
    for id_, annots in id2annots.items():
        model_id = ROBERTA if id_ < min_llama_input_id else LLAMA
        meth1 = metadata.loc[id_, "optionA"]
        meth2 = metadata.loc[id_, "optionB"]
        meth1_id, meth2_id = method_dict.get_idx_for_items([meth1, meth2])
        for a, b in annots:
            comps[model_id, meth1_id, meth2_id] += 1
            comps[model_id, meth2_id, meth1_id] += 1
            if a != b:
                winner, loser = (meth1_id, meth2_id) if a else (meth2_id, meth1_id)
                wins[model_id, loser, winner] += 1

    np.savez(output_dir / "counts.npz", comps=comps, wins=wins)
    method_dict.save(output_dir / "method.dict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute number of pairwise wins from HLV annotation outputs"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="directory containing .jsonl and metadata.csv files",
    )
    parser.add_argument("output_dir", type=Path, help="output directory")
    parser.add_argument(
        "-f", "--data-filename", default="data.jsonl", help="data JSONL filename"
    )
    parser.add_argument(
        "--min-llama-input-id",
        default=6240,
        type=int,
        help="min input ID produced with LLaMA",
    )
    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_dir,
        args.min_llama_input_id,
        args.data_filename,
    )
