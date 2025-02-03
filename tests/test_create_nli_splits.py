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

import functools
import json
import random
from pathlib import Path
from typing import Iterator

import pytest
from create_kfold_splits import main as main_kfold
from create_random_split import main as main_random


@pytest.fixture
def create_data(tmp_path):
    random.seed(0)

    def _create_data(
        n_unique_textpairs: int = 10, size: int = 100
    ) -> tuple[list[dict[str, dict]], Path]:
        unique_textpairs = [
            (
                " ".join("foo" for _ in range(i + 1)),
                " ".join("bar" for _ in range(i + 1)),
            )
            for i in range(n_unique_textpairs)
        ]
        data = []
        for prem, hyp in random.choices(unique_textpairs, k=size):
            data.append(
                {
                    "input": {"premise": prem, "hypothesis": hyp},
                    "output": {"verdict": random.choice("c e n".split())},
                }
            )
        with (tmp_path / "data.jsonl").open("w", encoding="utf8") as f:
            for obj in data:
                print(json.dumps(obj), file=f)

        return data, (tmp_path / "data.jsonl")

    return _create_data


@pytest.fixture
def output_dir(tmp_path):
    (tmp_path / "output").mkdir()
    return tmp_path / "output"


@pytest.fixture
def get_all_fold_textpairs(get_textpairs, output_dir):
    def _get_all_fold_textpairs(
        n_folds: int, set_: str, cast=set
    ) -> list[set[tuple[str, str]]]:
        return [
            cast(get_textpairs(output_dir / str(i) / f"{set_}.jsonl"))
            for i in range(n_folds)
        ]

    return _get_all_fold_textpairs


@pytest.fixture
def get_textpairs():
    def _get_textpairs(data_file: Path) -> Iterator[tuple[str, str]]:
        with data_file.open(encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                yield obj["input"]["premise"], obj["input"]["hypothesis"]

    return _get_textpairs


class TestKFold:
    def test_test_sets_are_partitions(
        self, create_data, output_dir, get_all_fold_textpairs
    ):
        data, path = create_data(n_unique_textpairs=10, size=100)
        n_folds = 3

        main_kfold(path, output_dir, n_folds)

        tst_textpairs_ls = get_all_fold_textpairs(n_folds, "test")
        for i in range(len(tst_textpairs_ls)):
            for j in range(i + 1, len(tst_textpairs_ls)):
                assert not (tst_textpairs_ls[i] & tst_textpairs_ls[j])
        assert functools.reduce(
            lambda x, y: x | y, tst_textpairs_ls
        ) == self._get_unique_textpairs(data)

    def test_equally_sized_test_sets(
        self, create_data, output_dir, get_all_fold_textpairs
    ):
        data, path = create_data(n_unique_textpairs=10, size=100)
        n_folds = 3

        main_kfold(path, output_dir, n_folds)

        tst_textpairs_ls = get_all_fold_textpairs(n_folds, "test")
        for i in range(len(self._get_unique_textpairs(data)) % n_folds):
            assert (
                len(tst_textpairs_ls[i])
                == len(self._get_unique_textpairs(data)) // n_folds + 1
            )
        for i in range(len(self._get_unique_textpairs(data)) % n_folds, n_folds):
            assert (
                len(tst_textpairs_ls[i])
                == len(self._get_unique_textpairs(data)) // n_folds
            )

    def test_train_test_sets_are_partitions(
        self, create_data, output_dir, get_all_fold_textpairs
    ):
        data, path = create_data(n_unique_textpairs=10, size=100)
        n_folds = 3

        main_kfold(path, output_dir, n_folds)

        trn_textpairs_ls = get_all_fold_textpairs(n_folds, "train")
        tst_textpairs_ls = get_all_fold_textpairs(n_folds, "test")
        all_textpairs = self._get_unique_textpairs(data)
        for trn_textpairs, tst_textpairs in zip(trn_textpairs_ls, tst_textpairs_ls):
            assert trn_textpairs | tst_textpairs == all_textpairs
            assert not (trn_textpairs & tst_textpairs)

    def test_with_shuffling(self, create_data, output_dir, get_all_fold_textpairs):
        data, path = create_data(n_unique_textpairs=10, size=10)
        data.sort(
            key=lambda obj: (
                obj["input"]["premise"],
                obj["input"]["hypothesis"],
                obj["output"]["verdict"],
            )
        )

        main_kfold(path, output_dir, fold=2, shuffle=True)

        assert [
            x
            for textpairs in get_all_fold_textpairs(n_folds=2, set_="test", cast=list)
            for x in textpairs
        ] != [(obj["input"]["premise"], obj["input"]["hypothesis"]) for obj in data]

    @staticmethod
    def _get_unique_textpairs(data: list[dict[str, dict]]) -> set[tuple[str, str]]:
        return set(
            (obj["input"]["premise"], obj["input"]["hypothesis"]) for obj in data
        )


@pytest.mark.parametrize("test_split_name", ["dev", "test"])
def test_random_split_partitions(
    create_data, output_dir, test_split_name, get_textpairs
):
    data, path = create_data()

    main_random(path, output_dir, test_split_name)
    trn_textpairs = set(get_textpairs(output_dir / "train.jsonl"))
    tst_textpairs = set(get_textpairs(output_dir / f"{test_split_name}.jsonl"))
    all_textpairs = set(
        (obj["input"]["premise"], obj["input"]["hypothesis"]) for obj in data
    )

    assert (trn_textpairs | tst_textpairs) == all_textpairs
    assert not (trn_textpairs & tst_textpairs)


def test_random_split_specify_portion(create_data, output_dir, get_textpairs):
    data, path = create_data()

    main_random(path, output_dir, test_portion=0.4)
    tst_textpairs = set(get_textpairs(output_dir / "test.jsonl"))
    all_textpairs = set(
        (obj["input"]["premise"], obj["input"]["hypothesis"]) for obj in data
    )

    assert len(tst_textpairs) / len(all_textpairs) == pytest.approx(0.4)
