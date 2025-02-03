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
from pathlib import Path


def main(
    file_: Path,
    output_dir: Path,
    test_split_name: str = "test",
    test_portion: float = 0.1,
    shuffle: bool = False,
) -> None:
    with file_.open(encoding="utf8") as f:
        data = list(map(json.loads, f))

    try:
        ids = list(set(map(_get_textpair, data)))
    except KeyError:
        ids = list(set(map(_get_text, data)))  # type: ignore[arg-type]
        get_id = _get_text
    else:
        get_id = _get_textpair  # type: ignore[assignment]
    if shuffle:
        random.shuffle(ids)
    tst_size = round(test_portion * len(ids))
    tst_ids = set(ids[:tst_size])

    trn_data, tst_data = [], []
    for obj in data:
        if get_id(obj) in tst_ids:
            tst_data.append(obj)
        else:
            trn_data.append(obj)

    with (output_dir / "train.jsonl").open("w", encoding="utf8") as f:
        for obj in trn_data:
            print(json.dumps(obj), file=f)
    with (output_dir / f"{test_split_name}.jsonl").open("w", encoding="utf8") as f:
        for obj in tst_data:
            print(json.dumps(obj), file=f)


def _get_textpair(obj: dict[str, dict]) -> tuple[str, str]:
    return obj["input"]["premise"], obj["input"]["hypothesis"]


def _get_text(obj: dict[str, dict]) -> str:
    return obj["input"]["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a random split.")
    parser.add_argument("file", type=Path, help="dataset file in our JSONL format")
    parser.add_argument("output_dir", type=Path, help="output directory")
    parser.add_argument(
        "--shuffle", action="store_true", help="shuffle before splitting"
    )
    parser.add_argument(
        "--test-split-name", default="test", help="name of the test split"
    )
    parser.add_argument("--test-portion", type=float, default=0.1, help="test portion")
    args = parser.parse_args()
    main(
        args.file,
        args.output_dir,
        args.test_split_name,
        args.test_portion,
        args.shuffle,
    )
