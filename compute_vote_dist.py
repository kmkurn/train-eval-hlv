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
from datetime import datetime
from pathlib import Path

import pandas as pd


def main(
    input_dir: Path,
    output_file: Path,
    data_filename: str = "data.jsonl",
    inputs_filename: str = "inputs.jsonl",
    min_llama_input_id: int = 1950,
) -> None:
    id2objs = defaultdict(list)
    with (input_dir / data_filename).open(encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            id2objs[obj["inputId"]].append(obj)
    id2text = {}
    with (input_dir / inputs_filename).open(encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            id2text[obj["id"]] = obj["text"]
    metadata = pd.read_csv(input_dir / "metadata.csv")
    metadata["model"] = metadata["id"].map(
        lambda i: "roberta" if i < min_llama_input_id else "llama"
    )

    metadata.groupby(["model", "method"]).apply(  # type: ignore[call-overload]
        create_vote_dist, id2objs, id2text, include_groups=False  # type: ignore[arg-type]
    ).dropna(subset="vote_dist").reset_index().to_pickle(output_file)


def create_vote_dist(
    df: pd.DataFrame,
    id2annotations: dict[int, list[dict]],
    id2text: dict[int, str],
) -> pd.Series:
    df.set_index
    votes = [0] * 4
    total = 0
    texts = set()
    for row in df.itertuples():
        id_ = row.id
        anns = id2annotations[id_]  # type: ignore[index]
        for ann in anns:
            ann["timestamp"] = datetime.fromisoformat(ann["timestamp"])
        annotator_ids = set()
        for ann in sorted(anns, key=lambda x: x["timestamp"], reverse=True):
            if ann["annotatorId"] not in annotator_ids:
                annotator_ids.add(ann["annotatorId"])
                pred = ann["preferredHLVAnnotation"]["isOptionAPreferred"]
                gold = ann["preferredHLVAnnotation"]["isOptionBPreferred"]
                if row.trueHLV == "A":
                    pred, gold = gold, pred
                votes[(0 if pred else 2) + (1 if gold else 0)] += 1
        total += len(annotator_ids)
        if annotator_ids:
            texts.add(id2text[id_])  # type: ignore[index]
    vote_dist = tuple([x / total for x in votes]) if total else None
    return pd.Series(
        {"vote_dist": vote_dist, "total": total, "nunique_texts": len(texts)}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute true vote dist. from HLV annotation outputs"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="directory containing .jsonl and metadata.csv files",
    )
    parser.add_argument("output_file", type=Path, help="output .pkl file")
    parser.add_argument(
        "-f", "--data-filename", default="data.jsonl", help="data JSONL filename"
    )
    parser.add_argument(
        "-i", "--inputs-filename", default="inputs.jsonl", help="inputs JSONL filename"
    )
    parser.add_argument(
        "--min-llama-input-id",
        default=1950,
        type=int,
        help="min input ID produced with LLaMA",
    )
    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_file,
        args.data_filename,
        args.inputs_filename,
        args.min_llama_input_id,
    )
