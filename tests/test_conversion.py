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

import csv
import json

import pandas as pd
import pytest
from convert_dataset import main


def test_chaosnli(tmp_path):
    data = [
        {
            "label_counter": {"c": 20, "e": 12, "n": 68},
            "example": {"premise": "foo bar", "hypothesis": "foo foo bar bar"},
        },
        {
            "label_counter": {"c": 48, "e": 45},
            "example": {"premise": "baz baz quux", "hypothesis": "baz quux"},
        },
    ]
    with (tmp_path / "chaosnli.jsonl").open("w", encoding="utf8") as f:
        for obj in data:
            print(json.dumps(obj), file=f)
    expected = [
        {
            "input": {"premise": "foo bar", "hypothesis": "foo foo bar bar"},
            "output": {"verdict": "c"},
        }
        for _ in range(20)
    ]
    expected.extend(
        [
            {
                "input": {"premise": "foo bar", "hypothesis": "foo foo bar bar"},
                "output": {"verdict": "e"},
            }
            for _ in range(12)
        ]
    )
    expected.extend(
        [
            {
                "input": {"premise": "foo bar", "hypothesis": "foo foo bar bar"},
                "output": {"verdict": "n"},
            }
            for _ in range(68)
        ]
    )
    expected.extend(
        [
            {
                "input": {"premise": "baz baz quux", "hypothesis": "baz quux"},
                "output": {"verdict": "c"},
            }
            for _ in range(48)
        ]
    )
    expected.extend(
        [
            {
                "input": {"premise": "baz baz quux", "hypothesis": "baz quux"},
                "output": {"verdict": "e"},
            }
            for _ in range(45)
        ]
    )

    result = list(main(tmp_path / "chaosnli.jsonl", dataset="chaosnli"))

    assert result == expected


def test_lewidi_not_convabuse(tmp_path, write_bc_jsonl):
    data = {
        "1": {"text": "foobar", "annotations": "1,1,0,1", "annotators": "A,B,C,D"},
        "2": {"text": "bazquux", "annotations": "0,0,0", "annotators": "A,C,D"},
    }
    (tmp_path / "lewidi.json").write_text(json.dumps(data), "utf8")
    write_bc_jsonl(
        [
            "foobar 1 A",
            "foobar 1 B",
            "foobar 0 C",
            "foobar 1 D",
            "bazquux 0 A",
            "bazquux 0 C",
            "bazquux 0 D",
        ],
        "lewidi.jsonl",
    )
    with (tmp_path / "lewidi.jsonl").open(encoding="utf8") as f:
        expected = list(map(json.loads, f))

    result = list(main(tmp_path / "lewidi.json", dataset="lewidi"))

    assert result == expected


def test_TAG_exports(tmp_path):
    pd.DataFrame(
        {
            "Sample_Text": [
                "foobar",
                "foobar",
                "foobar",
                "bar",
                "bar",
                "bar",
            ],
            "Type": [
                "overview",
                "highlight",
                "overview",
                "overview",
                "overview",
                "overview",
            ],
            "Area_of_Law": ["A", "Z", "B", "B", "A", "A"],
            "User_ID": [1, 1, 1, 1, 2, 1],
            "Certainty Rating": [73, None, 90, 8, 11, 78],
        }
    ).to_pickle(tmp_path / "data.pkl")
    expected = []
    for dat in ["foobar A B;1;0.73 0.9", "bar B A;1;0.08 0.78", "bar A;2;0.11"]:
        dat_, ann, conf = dat.split(";")
        splits = dat_.split()
        confs = [float(x) for x in conf.split()]
        expected.append(
            {
                "input": {"text": splits[0]},
                "output": {
                    "classes": splits[1:],
                    "confidences": [pytest.approx(x) for x in confs],
                },
                "metadata": {"annotator": int(ann)},
            }
        )

    result = list(main(tmp_path / "data.pkl", dataset="TAG"))

    assert result == expected


def test_mfrc(tmp_path):
    with (tmp_path / "final_mfrc_data.csv").open("w", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow("text subreddit bucket annotator annotation confidence".split())
        writer.writerow(["foo bar", "A1", "B1", "ann1", "X,Y", "C1"])
        writer.writerow(["foo bar", "A1", "B1", "ann2", "Z", "C1"])
        writer.writerow(["baz quux", "A2", "B2", "ann2", "Y,Z", "C2"])
    expected = [
        {
            "input": {"text": "foo bar"},
            "output": {"classes": ["X", "Y"]},
            "metadata": {"subreddit": "A1", "bucket": "B1", "annotator": "ann1"},
        },
        {
            "input": {"text": "foo bar"},
            "output": {"classes": ["Z"]},
            "metadata": {"subreddit": "A1", "bucket": "B1", "annotator": "ann2"},
        },
        {
            "input": {"text": "baz quux"},
            "output": {"classes": ["Y", "Z"]},
            "metadata": {"subreddit": "A2", "bucket": "B2", "annotator": "ann2"},
        },
    ]

    result = list(main(tmp_path / "final_mfrc_data.csv", dataset="MFRC"))

    assert result == expected
