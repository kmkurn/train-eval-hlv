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

from krippendorff_alpha import krippendorff_alpha, nominal_metric


def main(input_file: Path) -> float:
    with input_file.open(encoding="utf8") as f:
        objs = [json.loads(line) for line in f]
    for obj in objs:
        obj["timestamp"] = datetime.fromisoformat(obj["timestamp"])
    annotation: dict[int, dict[int, int]] = defaultdict(dict)
    for obj in sorted(objs, key=lambda o: o["timestamp"]):
        a = obj["preferredHLVAnnotation"]["isOptionAPreferred"]
        b = obj["preferredHLVAnnotation"]["isOptionBPreferred"]
        annotation[obj["annotatorId"]][obj["inputId"]] = (2 if a else 0) + (
            1 if b else 0
        )

    return krippendorff_alpha(list(annotation.values()), nominal_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute overall agreement of HLV annotation"
    )
    parser.add_argument(
        "input_file", type=Path, help=".jsonl file with the annotations"
    )
    args = parser.parse_args()
    res = main(args.input_file)
    print(res)
