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
import pickle
import textwrap

import numpy as np
import pytest
from create_hlv_annotation_input import main
from flair.data import Dictionary


@pytest.mark.parametrize("start_id", [0, 10])
def test_correct(tmp_path, start_id):
    true_hlv = np.array([[0.3, 0.4, 0.1], [0.1, 0.2, 0.9]])
    with (tmp_path / "method2hlv.npy").open("wb") as f:
        pickle.dump(
            {
                "a": np.stack(
                    [true_hlv, np.array([[0.1, 0.9, 0.8], [0.2, 0.9, 0.7]])], axis=0
                ),
                "b": np.stack(
                    [true_hlv, np.array([[0.9, 0.7, 0.8], [0.1, 0.9, 0.2]])], axis=0
                ),
            },
            file=f,
        )
    label_dict = Dictionary(add_unk=False)
    for l in "UVT":
        label_dict.add_item(l)
    label_dict.save(tmp_path / "label.dict")
    with (tmp_path / "inputs.jsonl").open("w", encoding="utf8") as f:
        print(json.dumps({"text": "foo bar"}), file=f)
        print(json.dumps({"text": "baz quux"}), file=f)

    main(
        tmp_path / "method2hlv.npy",
        tmp_path / "inputs.jsonl",
        tmp_path / "label.dict",
        tmp_path / "hlv-annotation",
        thresh=0.1,
        start_id=start_id,
    )

    expected = [
        {
            "id": start_id,
            "text": "foo bar",
            "optionA": [
                {"areaOfLaw": "V", "confidence": 0.4},
                {"areaOfLaw": "U", "confidence": 0.3},
            ],
            "optionB": [
                {"areaOfLaw": "V", "confidence": 0.9},
                {"areaOfLaw": "T", "confidence": 0.8},
            ],
        },
        {
            "id": start_id + 1,
            "text": "baz quux",
            "optionA": [
                {"areaOfLaw": "T", "confidence": 0.9},
                {"areaOfLaw": "V", "confidence": 0.2},
            ],
            "optionB": [
                {"areaOfLaw": "V", "confidence": 0.9},
                {"areaOfLaw": "T", "confidence": 0.7},
                {"areaOfLaw": "U", "confidence": 0.2},
            ],
        },
        {
            "id": start_id + 2,
            "text": "foo bar",
            "optionA": [
                {"areaOfLaw": "V", "confidence": 0.4},
                {"areaOfLaw": "U", "confidence": 0.3},
            ],
            "optionB": [
                {"areaOfLaw": "U", "confidence": 0.9},
                {"areaOfLaw": "T", "confidence": 0.8},
                {"areaOfLaw": "V", "confidence": 0.7},
            ],
        },
        {
            "id": start_id + 3,
            "text": "baz quux",
            "optionA": [
                {"areaOfLaw": "T", "confidence": 0.9},
                {"areaOfLaw": "V", "confidence": 0.2},
            ],
            "optionB": [
                {"areaOfLaw": "V", "confidence": 0.9},
                {"areaOfLaw": "T", "confidence": 0.2},
            ],
        },
    ]
    with (tmp_path / "hlv-annotation" / "input.jsonl").open(encoding="utf8") as f:
        assert [json.loads(l) for l in f] == expected
    assert (
        (tmp_path / "hlv-annotation" / "metadata.csv").read_text(encoding="utf8")
        == textwrap.dedent(
            f"""
    id,method,trueHLV
    {start_id},a,A
    {start_id+1},a,A
    {start_id+2},b,A
    {start_id+3},b,A
    """
        ).lstrip()
    )


def test_shuffle(tmp_path):
    true_hlv = np.random.rand(5, 3)
    with (tmp_path / "method2hlv.npy").open("wb") as f:
        pickle.dump(
            {
                meth: np.stack([true_hlv, np.random.rand(*true_hlv.shape)], axis=0)
                for meth in "ABCDEFG"
            },
            f,
        )
    label_dict = Dictionary(add_unk=False)
    for l in "UVT":
        label_dict.add_item(l)
    label_dict.save(tmp_path / "label.dict")
    with (tmp_path / "inputs.jsonl").open("w", encoding="utf8") as f:
        for i in range(true_hlv.shape[0]):
            print(json.dumps({"text": " ".join(["foo"] * (i + 1))}), file=f)

    main(
        tmp_path / "method2hlv.npy",
        tmp_path / "inputs.jsonl",
        tmp_path / "label.dict",
        tmp_path / "no-shuf",
        thresh=0.1,
    )
    main(
        tmp_path / "method2hlv.npy",
        tmp_path / "inputs.jsonl",
        tmp_path / "label.dict",
        tmp_path / "shuf",
        thresh=0.1,
        shuffle=True,
    )

    with (tmp_path / "no-shuf" / "input.jsonl").open(encoding="utf8") as f:
        no_shuf_data = [json.loads(l) for l in f]
    with (tmp_path / "shuf" / "input.jsonl").open(encoding="utf8") as f:
        shuf_data = [json.loads(l) for l in f]
    assert no_shuf_data != shuf_data
    assert no_shuf_data == sorted(shuf_data, key=lambda x: x["id"])


def test_subsample(tmp_path):
    true_hlv = np.random.rand(5, 3)
    with (tmp_path / "method2hlv.npy").open("wb") as f:
        pickle.dump(
            {
                meth: np.stack([true_hlv, np.random.rand(*true_hlv.shape)], axis=0)
                for meth in "ABCDEFG"
            },
            f,
        )
    label_dict = Dictionary(add_unk=False)
    for l in "UVT":
        label_dict.add_item(l)
    label_dict.save(tmp_path / "label.dict")
    texts = [" ".join(["foo"] * (i + 1)) for i in range(true_hlv.shape[0])]
    with (tmp_path / "inputs.jsonl").open("w", encoding="utf8") as f:
        for text in texts:
            print(json.dumps({"text": text}), file=f)

    main(
        tmp_path / "method2hlv.npy",
        tmp_path / "inputs.jsonl",
        tmp_path / "label.dict",
        tmp_path / "hlv-annotation",
        thresh=0.1,
        size=3,
    )

    with (tmp_path / "hlv-annotation" / "input.jsonl").open(encoding="utf8") as f:
        data = [json.loads(l) for l in f]
    assert len(data) == 21
    assert all(d["text"] in texts for d in data)


def test_randomise_options(tmp_path):
    true_hlv = np.random.rand(5, 3)
    with (tmp_path / "method2hlv.npy").open("wb") as f:
        pickle.dump(
            {
                meth: np.stack([true_hlv, np.random.rand(*true_hlv.shape)], axis=0)
                for meth in "ABCDEFG"
            },
            f,
        )
    label_dict = Dictionary(add_unk=False)
    for l in "UVT":
        label_dict.add_item(l)
    label_dict.save(tmp_path / "label.dict")
    with (tmp_path / "inputs.jsonl").open("w", encoding="utf8") as f:
        for i in range(true_hlv.shape[0]):
            print(json.dumps({"text": " ".join(["foo"] * (i + 1))}), file=f)

    main(
        tmp_path / "method2hlv.npy",
        tmp_path / "inputs.jsonl",
        tmp_path / "label.dict",
        tmp_path / "no-rand",
        thresh=0.1,
    )
    main(
        tmp_path / "method2hlv.npy",
        tmp_path / "inputs.jsonl",
        tmp_path / "label.dict",
        tmp_path / "rand",
        thresh=0.1,
        randomise_options=True,
    )
    with (tmp_path / "no-rand" / "input.jsonl").open(encoding="utf8") as f:
        no_rand_data = [json.loads(l) for l in f]
    with (tmp_path / "rand" / "input.jsonl").open(encoding="utf8") as f:
        rand_data = [json.loads(l) for l in f]
    assert no_rand_data != rand_data
    with (tmp_path / "rand" / "metadata.csv").open(encoding="utf8") as f:
        id2swapped = {
            int(row["id"]): row["trueHLV"] == "B" for row in csv.DictReader(f)
        }
    assert not all(id2swapped.values()) and any(id2swapped.values())
    assert no_rand_data == [
        {
            "id": dat["id"],
            "text": dat["text"],
            "optionA": dat["optionB"] if id2swapped[dat["id"]] else dat["optionA"],
            "optionB": dat["optionA"] if id2swapped[dat["id"]] else dat["optionB"],
        }
        for dat in rand_data
    ]
