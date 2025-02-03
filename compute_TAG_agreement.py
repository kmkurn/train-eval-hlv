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
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from krippendorff_alpha import krippendorff_alpha, nominal_metric


def main(input_file: Path, output_file: Path) -> None:
    with input_file.open(encoding="utf8") as f:
        data = [json.loads(l) for l in f]

    labels = set()
    for d in data:
        labels.update(d["output"]["classes"])
        d["output"]["classes"] = set(d["output"]["classes"])

    lab2alpha = {}
    for lab in tqdm(labels, unit="label", desc="Computing alpha"):
        annotation: dict[int, dict[str, bool]] = defaultdict(dict)
        for d in data:
            annotation[d["metadata"]["annotator"]][d["input"]["text"]] = (
                lab in d["output"]["classes"]
            )
        lab2alpha[lab] = krippendorff_alpha(list(annotation.values()), nominal_metric)

    with output_file.open("wb") as f:
        pickle.dump(lab2alpha, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute agreements of TAG data")
    parser.add_argument("input_file", type=Path, help="all.jsonl file")
    parser.add_argument("output_file", type=Path, help="output .pkl file")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
