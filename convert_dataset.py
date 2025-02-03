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
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd


def main(file_: Path, dataset: str) -> Iterator[dict[str, dict]]:
    if dataset == "chaosnli":
        yield from _read_chaosnli(file_)
    elif dataset == "lewidi":
        yield from _read_lewidi(file_)
    elif dataset == "TAG":
        yield from _read_TAG(file_)
    elif dataset == "MFRC":
        yield from _read_MFRC(file_)
    else:
        raise ValueError(f"unrecognised dataset: {dataset}")


def _read_chaosnli(file_: Path) -> Iterator[dict[str, dict]]:
    with file_.open(encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            for label, count in obj["label_counter"].items():
                for _ in range(count):
                    yield {
                        "input": {
                            "premise": obj["example"]["premise"],
                            "hypothesis": obj["example"]["hypothesis"],
                        },
                        "output": {"verdict": label},
                    }


def _read_lewidi(file_: Path) -> Iterator[dict[str, dict]]:
    data = json.loads(file_.read_text("utf8"))
    for obj in data.values():
        for lab, ann in zip(
            obj["annotations"].split(","), obj["annotators"].split(",")
        ):
            yield {
                "input": {"text": obj["text"]},
                "output": {"class": lab == "1"},
                "metadata": {"annotator": ann},
            }


def _read_TAG(file_: Path) -> Iterator[dict[str, dict]]:
    df = pd.read_pickle(file_)
    for row in (
        df[df.Type == "overview"]
        .groupby(["Sample_Text", "User_ID"], sort=False)
        .apply(_get_areas_and_ratings, include_groups=False)
        .itertuples()
    ):
        yield {
            "input": {"text": row.Index[0]},
            "output": {"classes": row.areas, "confidences": row.ratings},
            "metadata": {"annotator": row.Index[1]},
        }


def _read_MFRC(file_: Path) -> Iterator[dict[str, dict]]:
    id2values = defaultdict(list)
    with file_.open(encoding="utf8") as f:
        lines = list(f)[1:]  # skip header line
        reader = csv.reader(lines)
        for row in reader:
            text, subreddit, bucket, annotator, annotation, _ = row
            id2values[text].append((subreddit, bucket, annotator, annotation))
    for text, values in id2values.items():
        for subreddit, bucket, annotator, annotation in values:
            yield {
                "input": {"text": text},
                "output": {"classes": annotation.split(",")},
                "metadata": {
                    "subreddit": subreddit,
                    "bucket": bucket,
                    "annotator": annotator,
                },
            }


def _get_areas_and_ratings(df: pd.DataFrame) -> pd.Series:
    areas, ratings = [], []
    for _, row in df.drop_duplicates("Area_of_Law").iterrows():
        areas.append(row["Area_of_Law"])
        ratings.append(row["Certainty Rating"] / 100)
    return pd.Series([areas, ratings], index=["areas", "ratings"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to our JSONL format.")
    parser.add_argument("file", type=Path, help="data file to convert")
    parser.add_argument(
        "dataset", choices=["chaosnli", "lewidi", "TAG", "MFRC"], help="dataset name"
    )
    args = parser.parse_args()
    for obj in main(args.file, args.dataset):
        print(json.dumps(obj))
