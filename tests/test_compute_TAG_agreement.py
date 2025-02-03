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

import json
import pickle

from compute_TAG_agreement import main


def test_ok(tmp_path):
    data = [
        {
            "input": {"text": "foo"},
            "output": {"classes": ["A"]},
            "metadata": {"annotator": 1},
        },
        {
            "input": {"text": "foo"},
            "output": {"classes": ["A", "B"]},
            "metadata": {"annotator": 2},
        },
        {
            "input": {"text": "bar"},
            "output": {"classes": ["B"]},
            "metadata": {"annotator": 3},
        },
    ]
    with (tmp_path / "all.jsonl").open("w", encoding="utf8") as f:
        for d in data:
            print(json.dumps(d), file=f)

    main(tmp_path / "all.jsonl", tmp_path / "agreement.pkl")

    with (tmp_path / "agreement.pkl").open("rb") as f:
        res = pickle.load(f)
    assert set(res.keys()) == {"A", "B"}
    assert all(a <= 1 for a in res.values())
