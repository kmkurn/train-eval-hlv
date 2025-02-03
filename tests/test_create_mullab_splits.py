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
from collections import Counter
from pathlib import Path
from typing import Iterator

import pytest
from create_kfold_splits import main as main_kfold
from create_random_split import main as main_random


@pytest.fixture
def create_data(tmp_path):
    random.seed(0)

    def _create_data(n_unique_texts=10, size=100):
        unique_texts = [
            " ".join("foo" for _ in range(i + 1)) for i in range(n_unique_texts)
        ]
        data = []
        labelset = list("ABC")
        for text in random.choices(unique_texts, k=size):
            random.shuffle(labelset)
            data.append(
                {
                    "input": {"text": text},
                    "output": {"classes": labelset[: random.randint(0, 3)]},
                }
            )
        with (tmp_path / "data.jsonl").open("w", encoding="utf8") as f:
            for obj in data:
                print(json.dumps(obj), file=f)

        return data, tmp_path / "data.jsonl"

    return _create_data


@pytest.fixture
def output_dir(tmp_path):
    (tmp_path / "output").mkdir()
    return tmp_path / "output"


@pytest.fixture
def get_all_fold_texts(get_texts, output_dir):
    def _get_all_fold_texts(n_folds: int, set_: str, cast=set) -> list[set[str]]:
        return [
            cast(get_texts(output_dir / str(i) / f"{set_}.jsonl"))
            for i in range(n_folds)
        ]

    return _get_all_fold_texts


@pytest.fixture
def get_texts():
    def _get_texts(data_file: Path) -> Iterator[str]:
        with data_file.open(encoding="utf8") as f:
            for line in f:
                obj = json.loads(line)
                yield obj["input"]["text"]

    return _get_texts


class TestKFold:
    @pytest.mark.parametrize("stratified", [False, True])
    def test_test_sets_are_partitions(
        self, create_data, output_dir, get_all_fold_texts, stratified
    ):
        data, path = create_data(n_unique_texts=10, size=100)
        n_folds = 3

        self._main_kfold(path, output_dir, n_folds, stratified=stratified)

        tst_texts_ls = get_all_fold_texts(n_folds, "test")
        for i in range(len(tst_texts_ls)):
            for j in range(i + 1, len(tst_texts_ls)):
                assert not (tst_texts_ls[i] & tst_texts_ls[j])
        assert functools.reduce(
            lambda x, y: x | y, tst_texts_ls
        ) == self._get_unique_texts(data)

    @pytest.mark.parametrize("stratified", [False, True])
    def test_equally_sized_test_sets(
        self, create_data, output_dir, get_all_fold_texts, stratified
    ):
        data, path = create_data(n_unique_texts=10, size=100)
        n_folds = 3

        self._main_kfold(path, output_dir, n_folds, stratified=stratified)

        tst_texts_ls = get_all_fold_texts(n_folds, "test")
        for i in range(len(self._get_unique_texts(data)) % n_folds):
            assert (
                len(tst_texts_ls[i]) == len(self._get_unique_texts(data)) // n_folds + 1
            )
        for i in range(len(self._get_unique_texts(data)) % n_folds, n_folds):
            assert len(tst_texts_ls[i]) == len(self._get_unique_texts(data)) // n_folds

    @pytest.mark.parametrize("stratified", [False, True])
    def test_train_test_sets_are_partitions(
        self, create_data, output_dir, get_all_fold_texts, stratified
    ):
        data, path = create_data(n_unique_texts=10, size=100)
        n_folds = 3

        self._main_kfold(path, output_dir, n_folds, stratified=stratified)

        trn_texts_ls = get_all_fold_texts(n_folds, "train")
        tst_texts_ls = get_all_fold_texts(n_folds, "test")
        all_texts = self._get_unique_texts(data)
        for trn_texts, tst_texts in zip(trn_texts_ls, tst_texts_ls):
            assert trn_texts | tst_texts == all_texts
            assert not (trn_texts & tst_texts)

    @pytest.mark.parametrize("stratified", [False, True])
    def test_with_shuffling(
        self, create_data, output_dir, get_all_fold_texts, stratified
    ):
        data, path = create_data(n_unique_texts=10, size=10)
        data.sort(
            key=lambda obj: (
                obj["input"]["text"],
                obj["output"]["classes"],
            )
        )

        self._main_kfold(path, output_dir, fold=2, shuffle=True, stratified=stratified)

        assert [
            x
            for texts in get_all_fold_texts(n_folds=2, set_="test", cast=list)
            for x in texts
        ] != [obj["input"]["text"] for obj in data]

    def test_stratified(self, create_data, output_dir, get_all_fold_texts):
        data, path = create_data(n_unique_texts=10, size=20000)
        n_folds = 3
        label_freq = Counter(tuple(sorted(obj["output"]["classes"])) for obj in data)
        sum_freq = sum(label_freq.values())

        self._main_kfold(path, output_dir, n_folds, stratified=True)

        for i in range(n_folds):
            tst_lab_frq = Counter()
            with (output_dir / str(i) / "test.jsonl").open(encoding="utf8") as f:
                for line in f:
                    obj = json.loads(line)
                    tst_lab_frq[tuple(sorted(obj["output"]["classes"]))] += 1
            tst_sum_frq = sum(tst_lab_frq.values())
            for lab, frq in label_freq.items():
                assert frq / sum_freq == pytest.approx(
                    tst_lab_frq[lab] / tst_sum_frq, abs=1e-2
                )
            for lab, frq in tst_lab_frq.items():
                assert frq / tst_sum_frq == pytest.approx(
                    label_freq[lab] / sum_freq, abs=1e-2
                )

    @staticmethod
    def _main_kfold(*args, **kwargs):
        kwargs["is_nli"] = False
        return main_kfold(*args, **kwargs)

    @staticmethod
    def _get_unique_texts(data: list[dict[str, dict]]) -> set[str]:
        return set(obj["input"]["text"] for obj in data)


@pytest.mark.parametrize("test_split_name", ["dev", "test"])
def test_random_split_partitions(create_data, output_dir, test_split_name, get_texts):
    data, path = create_data()

    main_random(path, output_dir, test_split_name)
    trn_texts = set(get_texts(output_dir / "train.jsonl"))
    tst_texts = set(get_texts(output_dir / f"{test_split_name}.jsonl"))
    all_texts = set(obj["input"]["text"] for obj in data)

    assert (trn_texts | tst_texts) == all_texts
    assert not (trn_texts & tst_texts)


def test_random_split_specify_portion(create_data, output_dir, get_texts):
    data, path = create_data()

    main_random(path, output_dir, test_portion=0.4)
    tst_texts = set(get_texts(output_dir / "test.jsonl"))
    all_texts = set(obj["input"]["text"] for obj in data)

    assert len(tst_texts) / len(all_texts) == pytest.approx(0.4)
