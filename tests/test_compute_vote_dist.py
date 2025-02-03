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
import textwrap

import pandas as pd
import pytest
from compute_vote_dist import main


def test_correct(tmp_path):
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
        {
            "inputId": 89,
            "timestamp": "2024-09-30T22:44:00+00:00",
            "annotatorId": 214,
            "areasOfLawAnnotation": {"selectedAreas": ["B"]},
            "preferredHLVAnnotation": {
                "isOptionAPreferred": False,
                "isOptionBPreferred": True,
                "confidence": 0.75,
            },
        },
        {
            "inputId": 4646,
            "timestamp": "2024-09-30T22:44:00+00:00",
            "annotatorId": 214,
            "areasOfLawAnnotation": {"selectedAreas": ["A", "B"]},
            "preferredHLVAnnotation": {
                "isOptionAPreferred": False,
                "isOptionBPreferred": False,
                "confidence": 1,
            },
        },
    ]
    with (tmp_path / "data.jsonl").open("w", encoding="utf8") as f:
        for dat in data:
            print(json.dumps(dat), file=f)
    data = [
        {
            "id": 4646,
            "text": "foo",
        },
        {
            "id": 89,
            "text": "foo",
        },
        {
            "id": 4121,
            "text": "bar",
        },
        {
            "id": 70,
            "text": "foo",
        },
    ]
    with (tmp_path / "inputs.jsonl").open("w", encoding="utf8") as f:
        for dat in data:
            print(json.dumps(dat), file=f)
    with (tmp_path / "metadata.csv").open("w", encoding="utf8") as f:
        print(
            textwrap.dedent(
                """
        id,method,trueHLV
        4646,M2,B
        89,M1,A
        70,M1,B
        4121,M2,A
        81,M3,A
        """
            ),
            file=f,
        )

    main(tmp_path, tmp_path / "vote_dists.pkl")

    df = pd.read_pickle(tmp_path / "vote_dists.pkl")
    assert df.to_dict(orient="list") == {
        "model": ["llama", "roberta"],
        "method": ["M2", "M1"],
        "total": [2, 3],
        "nunique_texts": [2, 1],
        "vote_dist": [
            pytest.approx((0, 1 / 2, 1 / 2, 0)),
            pytest.approx((2 / 3, 0, 1 / 3, 0)),
        ],
    }
