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

from compute_hlv_annotation_agreement import main


def test_ok(tmp_path):
    data = [
        {
            "inputId": 70,
            "timestamp": "2024-09-27T04:49:00+00:00",
            "annotatorId": 214,
            "areasOfLawAnnotation": {"selectedAreas": ["A"]},
            "preferredHLVAnnotation": {
                "isOptionAPreferred": False,
                "isOptionBPreferred": False,
                "confidence": 0.75,
            },
        },
        {
            "inputId": 70,
            "timestamp": "2024-09-30T22:47:00+00:00",
            "annotatorId": 262,
            "areasOfLawAnnotation": {"selectedAreas": ["A", "B"]},
            "preferredHLVAnnotation": {
                "isOptionAPreferred": True,
                "isOptionBPreferred": False,
                "confidence": 1,
            },
        },
        {
            "inputId": 70,
            "timestamp": "2024-09-30T22:44:00+00:00",
            "annotatorId": 262,
            "areasOfLawAnnotation": {"selectedAreas": ["A", "B"]},
            "preferredHLVAnnotation": {
                "isOptionAPreferred": False,
                "isOptionBPreferred": False,
                "confidence": 1,
            },
        },
        {
            "inputId": 4121,
            "timestamp": "2024-09-30T22:44:00+00:00",
            "annotatorId": 214,
            "areasOfLawAnnotation": {"selectedAreas": ["X"]},
            "preferredHLVAnnotation": {
                "isOptionAPreferred": True,
                "isOptionBPreferred": True,
                "confidence": 0.5,
            },
        },
    ]
    with (tmp_path / "data.jsonl").open("w", encoding="utf8") as f:
        for dat in data:
            print(json.dumps(dat), file=f)

    res = main(tmp_path / "data.jsonl")

    assert res <= 1
