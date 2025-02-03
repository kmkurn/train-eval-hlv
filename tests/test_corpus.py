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

import pytest
from corpus import (
    AmbiguouslyLabelledTrainSetCorpus,
    AnnotatorLabelledTrainSetCorpus,
    AnnotatorRankingScoredTrainSetCorpus,
    MajorityVotedTrainSetCorpus,
    SoftLabelledAndMajorityVotedTrainSetCorpus,
    SoftLabelledTrainSetCorpus,
)


def test_nli_mv(tmp_path, write_nli_jsonl):
    data = [
        "foo baz E",
        "foo bar E",
        "bar baz E",
        "bar baz N",
        "foo baz N",
        "bar baz N",
    ]
    write_nli_jsonl(data, "train.jsonl")

    corpus = MajorityVotedTrainSetCorpus(tmp_path, labels=("E", "N"))

    assert len(corpus.train) == 3
    for dat, dp in zip(["foo baz E", "foo bar E", "bar baz N"], corpus.train):
        p, h, v = dat.split()
        assert dp.first.text == p
        assert dp.second.text == h
        assert dp.get_label(corpus.hard_label_type).value == v


def test_bc_mv(tmp_path, write_bc_jsonl):
    data = [
        "foobaz 1",
        "foobar 1",
        "barbaz 1",
        "barbaz 0",
        "foobaz 0",
        "barbaz 0",
    ]
    write_bc_jsonl(data, "train.jsonl")

    corpus = MajorityVotedTrainSetCorpus(tmp_path, labels=("True", "False"))

    assert len(corpus.train) == 3
    for dat, dp in zip(["foobaz 1", "foobar 1", "barbaz 0"], corpus.train):
        t, v = dat.split()
        assert dp.get_metadata("orig_text") == t
        assert dp.get_label(corpus.hard_label_type).value == str(v == "1")


def test_multilabel_mv(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A;", "foo B;", "foo A B;"], "train.jsonl")

    corpus = MajorityVotedTrainSetCorpus(tmp_path, labels=("A", "B"))

    assert len(corpus.train) == 1
    assert corpus.train[0].get_metadata("orig_text") == "foo"
    assert len(corpus.train[0].get_labels(corpus.hard_label_type)) == 2


def test_nli_ambiguous_labelling(tmp_path, write_nli_jsonl):
    write_nli_jsonl(["foo foo E", "bar bar N", "foo foo C", "foo foo E"], "train.jsonl")

    corpus = AmbiguouslyLabelledTrainSetCorpus(tmp_path, labels=("E", "N", "C"))

    assert len(corpus.train) == 3
    for dat, dp in zip(["foo foo E", "foo foo C", "bar bar N"], corpus.train):
        p, h, v = dat.split()
        assert dp.first.text == p
        assert dp.second.text == h
        assert dp.get_label(corpus.hard_label_type).value == v


def test_bc_ambiguous_labelling(tmp_path, write_bc_jsonl):
    write_bc_jsonl(["foofoo 1", "barbar 0", "foofoo 0", "foofoo 1"], "train.jsonl")

    corpus = AmbiguouslyLabelledTrainSetCorpus(tmp_path, labels=("True", "False"))

    assert len(corpus.train) == 3
    for dat, dp in zip(["foofoo 1", "foofoo 0", "barbar 0"], corpus.train):
        t, v = dat.split()
        assert dp.get_metadata("orig_text")
        assert dp.get_label(corpus.hard_label_type).value == str(v == "1")


def test_nli_soft_labelling(tmp_path, write_nli_jsonl):
    write_nli_jsonl(["foo foo E", "bar bar N", "foo foo N", "foo foo E"], "train.jsonl")

    corpus = SoftLabelledTrainSetCorpus(tmp_path, labels=("N", "E"))

    assert len(corpus.train) == 2
    assert corpus.train[0].first.text == "foo"
    assert corpus.train[0].second.text == "foo"
    assert {
        l.value: l.score for l in corpus.train[0].get_labels(corpus.soft_label_type)
    } == pytest.approx({"E": 2 / 3, "N": 1 / 3})
    assert not corpus.train[0].get_labels(corpus.hard_label_type)
    assert corpus.train[1].first.text == "bar"
    assert corpus.train[1].second.text == "bar"
    assert {
        l.value: l.score for l in corpus.train[1].get_labels(corpus.soft_label_type)
    } == pytest.approx({"E": 0, "N": 1})
    assert not corpus.train[1].get_labels(corpus.hard_label_type)


def test_bc_soft_labelling(tmp_path, write_bc_jsonl):
    write_bc_jsonl(["foofoo 1", "barbar 0", "foofoo 0", "foofoo 1"], "train.jsonl")

    corpus = SoftLabelledTrainSetCorpus(tmp_path, labels=("False", "True"))

    assert len(corpus.train) == 2
    assert corpus.train[0].get_metadata("orig_text") == "foofoo"
    assert {
        l.value: l.score for l in corpus.train[0].get_labels(corpus.soft_label_type)
    } == pytest.approx({"True": 2 / 3, "False": 1 / 3})
    assert not corpus.train[0].get_labels(corpus.hard_label_type)
    assert corpus.train[1].get_metadata("orig_text") == "barbar"
    assert {
        l.value: l.score for l in corpus.train[1].get_labels(corpus.soft_label_type)
    } == pytest.approx({"True": 0, "False": 1})
    assert not corpus.train[1].get_labels(corpus.hard_label_type)


def test_nli_soft_labelling_and_mv(tmp_path, write_nli_jsonl):
    data = [
        "foo baz E",
        "foo bar E",
        "bar baz E",
        "bar baz N",
        "foo baz N",
        "bar baz N",
    ]
    write_nli_jsonl(data, "train.jsonl")

    corpus = SoftLabelledAndMajorityVotedTrainSetCorpus(tmp_path, labels=("E", "N"))

    assert {
        (dp.first.text, dp.second.text): dp.get_label(corpus.hard_label_type).value
        for dp in corpus.train
    } == {("foo", "baz"): "E", ("foo", "bar"): "E", ("bar", "baz"): "N"}
    assert {
        (dp.first.text, dp.second.text): {
            l.value: l.score for l in dp.get_labels(corpus.soft_label_type)
        }
        for dp in corpus.train
    } == {
        ("foo", "baz"): pytest.approx({"E": 0.5, "N": 0.5}),
        ("foo", "bar"): pytest.approx({"E": 1.0, "N": 0.0}),
        ("bar", "baz"): pytest.approx({"E": 1 / 3, "N": 2 / 3}),
    }
    assert all(
        {l.value for l in dp.get_labels("multitask_id")}
        == {corpus.soft_label_type, corpus.hard_label_type}
        for dp in corpus.train
    )


def test_bc_soft_labelling_and_mv(tmp_path, write_bc_jsonl):
    data = [
        "foobaz 1",
        "foobar 1",
        "barbaz 1",
        "barbaz 0",
        "foobaz 0",
        "barbaz 0",
    ]
    write_bc_jsonl(data, "train.jsonl")

    corpus = SoftLabelledAndMajorityVotedTrainSetCorpus(
        tmp_path, labels=("True", "False")
    )

    assert {
        dp.get_metadata("orig_text"): dp.get_label(corpus.hard_label_type).value
        for dp in corpus.train
    } == {"foobaz": "True", "foobar": "True", "barbaz": "False"}
    assert {
        dp.get_metadata("orig_text"): {
            l.value: l.score for l in dp.get_labels(corpus.soft_label_type)
        }
        for dp in corpus.train
    } == {
        "foobaz": pytest.approx({"True": 0.5, "False": 0.5}),
        "foobar": pytest.approx({"True": 1.0, "False": 0.0}),
        "barbaz": pytest.approx({"True": 1 / 3, "False": 2 / 3}),
    }
    assert all(
        {l.value for l in dp.get_labels("multitask_id")}
        == {corpus.soft_label_type, corpus.hard_label_type}
        for dp in corpus.train
    )


def test_multilabel_soft_labelling_and_mv(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A;", "foo B;", "foo A B;"], "train.jsonl")

    corpus = SoftLabelledAndMajorityVotedTrainSetCorpus(tmp_path, labels=("A", "B"))

    assert len(corpus.train) == 1
    assert corpus.train[0].get_metadata("orig_text") == "foo"
    assert {l.value for l in corpus.train[0].get_labels(corpus.hard_label_type)} == {
        "A",
        "B",
    }


def test_nli_per_annotator_labelling(tmp_path, write_nli_jsonl):
    write_nli_jsonl(
        ["foo bar E 1", "bar bar E 2", "foo bar N 2", "bar bar N 1", "baz quux N 3"],
        "train.jsonl",
    )
    expected = {
        ("foo", "bar"): {"1": "E", "2": "N"},
        ("bar", "bar"): {"2": "E", "1": "N"},
        ("baz", "quux"): {"3": "N"},
    }

    corpus = AnnotatorLabelledTrainSetCorpus(tmp_path, labels=("E", "N"))

    for dp in corpus.train:
        for ann, lab in expected[dp.first.text, dp.second.text].items():
            assert (
                dp.get_label(f"{corpus.hard_label_type} by annotator {ann}").value
                == lab
            )


def test_bc_per_annotator_labelling(tmp_path, write_bc_jsonl):
    write_bc_jsonl(
        ["foobar 1 A", "barbar 1 B", "foobar 0 B", "barbar 0 A", "bazquux 0 C"],
        "train.jsonl",
    )
    expected = {
        "foobar": {"A": "True", "B": "False"},
        "barbar": {"B": "True", "A": "False"},
        "bazquux": {"C": "False"},
    }

    corpus = AnnotatorLabelledTrainSetCorpus(tmp_path, labels=("True", "False"))

    for dp in corpus.train:
        for ann, lab in expected[dp.get_metadata("orig_text")].items():
            assert (
                dp.get_label(f"{corpus.hard_label_type} by annotator {ann}").value
                == lab
            )


def test_multilabel_per_annotator_labelling(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A B;1", "foo B;2"], "train.jsonl")

    corpus = AnnotatorLabelledTrainSetCorpus(tmp_path, labels=("A", "B"))

    assert len(corpus.train) == 1
    assert {
        l.value
        for l in corpus.train[0].get_labels(f"{corpus.hard_label_type} by annotator 1")
    } == {"A", "B"}


def test_bc_hard_annotator_ranking_scores(tmp_path, write_bc_jsonl):
    write_bc_jsonl(
        ["foo 1 A", "foo 0 B", "foo 1 C", "bar 0 A", "bar 1 C", "baz 1 D"],
        "train.jsonl",
    )
    expected = {
        "foo": ("True", 1.5),
        "bar": ("False", 1.5),
        "baz": ("True", 1),
    }

    corpus = AnnotatorRankingScoredTrainSetCorpus(tmp_path, labels=("False", "True"))

    for dp in corpus.train:
        lab, weight = expected[dp.get_metadata("orig_text")]
        l = dp.get_label(corpus.hard_label_type)
        assert l.value == lab
        assert l.score == pytest.approx(weight)


def test_multilabel_hard_annotator_ranking_scores(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(
        ["foo A B;1", "foo;2", "foo A;3", "bar B;1", "bar A;3", "baz A B;4"],
        "train.jsonl",
    )
    expected = {
        "foo": ({"A"}, (1.5 + 2) / 2),
        "bar": (set(), (1.5 + 1) / 2),
        "baz": ({"A", "B"}, (1 + 1) / 2),
    }

    corpus = AnnotatorRankingScoredTrainSetCorpus(tmp_path, labels=("A", "B"))

    for dp in corpus.train:
        labs, weight = expected[dp.get_metadata("orig_text")]
        assert {l.value for l in dp.get_labels(corpus.hard_label_type)} == labs
        assert dp.get_metadata("weight") == pytest.approx(weight)


def test_bc_soft_annotator_ranking_scores(tmp_path, write_bc_jsonl):
    write_bc_jsonl(
        ["foo 1 A", "foo 0 B", "foo 1 C", "bar 0 A", "bar 1 C", "baz 1 D"],
        "train.jsonl",
    )
    expected = {
        "foo": ("True", (2 / 3 + 1 / 2) / 2 + 0 + 2 / 3),
        "bar": ("False", (2 / 3 + 1 / 2) / 2 + 2 / 3),
        "baz": ("True", 1),
    }

    corpus = AnnotatorRankingScoredTrainSetCorpus(
        tmp_path, labels=("False", "True"), soft_scoring=True
    )

    for dp in corpus.train:
        lab, weight = expected[dp.get_metadata("orig_text")]
        l = dp.get_label(corpus.hard_label_type)
        assert l.value == lab
        assert l.score == pytest.approx(weight)


def test_multilabel_soft_annotator_ranking_scores(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(
        ["foo A B;1", "foo;2", "foo A;3", "bar B;1", "bar A;3", "baz A B;4"],
        "train.jsonl",
    )
    expected = {
        "foo": (
            {"A"},
            ((2 / 3 + 1 / 2) / 2 + 0 + 2 / 3 + 0 + 2 / 3 + (2 / 3 + 1 / 2) / 2) / 2,
        ),
        "bar": (set(), ((2 / 3 + 1 / 2) / 2 + 2 / 3 + 0 + (2 / 3 + 1 / 2) / 2) / 2),
        "baz": ({"A", "B"}, (1 + 1) / 2),
    }

    corpus = AnnotatorRankingScoredTrainSetCorpus(
        tmp_path, labels=("A", "B"), soft_scoring=True
    )

    for dp in corpus.train:
        labs, weight = expected[dp.get_metadata("orig_text")]
        assert {l.value for l in dp.get_labels(corpus.hard_label_type)} == labs
        assert dp.get_metadata("weight") == pytest.approx(weight)


def test_annotator_ranking_scored_corpus_with_max_length(tmp_path, write_mullab_jsonl):
    write_mullab_jsonl(["foo A;1", "bar B;2"], "train.jsonl")

    AnnotatorRankingScoredTrainSetCorpus(tmp_path, labels=("A", "B"), max_length=1000)
