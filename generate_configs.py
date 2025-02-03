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
import textwrap
from pathlib import Path


class ConfigGenerator:
    def __init__(self, lr=(-7, -3), batch_size=(0, 12)):
        self._lr_bounds = lr
        self._bsz_candidates = list(range(*batch_size))

    def generate(self):
        return {
            "lr": 10 ** random.uniform(*self._lr_bounds),
            "batch_size": 2 ** random.choice(self._bsz_candidates),
        }

    @property
    def description(self):
        return textwrap.dedent(
            f"""
        Distributions:
        - log10(lr) ~ uniform({', '.join(str(x) for x in self._lr_bounds)})
        - log2(batch_size) ~ choice({self._bsz_candidates})
        """
        ).lstrip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training configs.")
    parser.add_argument(
        "output_dir", type=Path, help="save configs under this directory"
    )
    parser.add_argument(
        "--lr", type=float, default=[-7, -3], nargs=2, help="bounds for log10(lr)"
    )
    parser.add_argument(
        "--bsz", type=int, default=[2, 12], nargs=2, help="bounds for log2(batch_size)"
    )
    parser.add_argument(
        "-n", "--size", type=int, default=1, help="generate this many times"
    )
    args = parser.parse_args()
    gen = ConfigGenerator(args.lr, args.bsz)
    (args.output_dir / "README.txt").write_text(gen.description, "utf8")
    for i in range(1, args.size + 1):
        with (args.output_dir / f"config_{i}.json").open("w", encoding="utf8") as f:
            json.dump(gen.generate(), f, indent=2, sort_keys=True)
