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

import abc
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union, overload

from flair.data import Corpus, DataPair, Dictionary, Label, Sentence, TextPair
from flair.datasets import FlairDatapointDataset
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier, TextPairClassifier
from flair.nn import Classifier
from tqdm import tqdm

from classifiers import (
    AnnotatorEnsembleTextClassifier,
    AnnotatorEnsembleTextPairClassifier,
    InstWeightingTextClassifier,
    LossAggregationTextClassifier,
    LossAggregationTextPairClassifier,
    SingleEvaluationTaskMultitaskModel,
    TextPairSoftClassifier,
    TextSoftClassifier,
)


class NotNLICorpusError(Exception):
    pass


class HLVCorpus(Corpus[Union["TCDataPoint", "NLIDataPoint"]]):
    _nli_hard_label_type = "hard verdict"
    _bc_hard_label_type = "hard class"

    def __init__(
        self,
        data_dir: Path,
        labels: Iterable[str],
        sort_train_desc: bool = False,
        max_length: int = 1_000_000_000,
    ) -> None:
        self.__labels = labels
        self.__max_length = max_length
        DatasetT = (
            FlairDatapointDataset[TCDataPoint] | FlairDatapointDataset[NLIDataPoint]
        )
        dat: DatasetT
        dev_dat: Optional[DatasetT]
        tst_dat: Optional[DatasetT]
        try:
            dps = list(self._read_nli_file(data_dir / "train.jsonl"))
        except NotNLICorpusError:
            self._is_nli = False
            self._mullab: Optional[bool] = None
            dat = FlairDatapointDataset(
                list(
                    self.preprocess_train_datapoints(
                        self._read_tc_file(data_dir / "train.jsonl")
                    )
                )
            )
            try:
                dev_sents = list(
                    self._compute_hard_labels(  # needed to avoid training error with flair
                        self.compute_hlv(self._read_tc_file(data_dir / "dev.jsonl"))
                    )
                )
            except FileNotFoundError:
                dev_dat = None
            else:
                dev_dat = FlairDatapointDataset(dev_sents)
            try:
                tst_sents = list(
                    self.compute_hlv(self._read_tc_file(data_dir / "test.jsonl"))
                )
            except FileNotFoundError:
                tst_dat = None
            else:
                tst_dat = FlairDatapointDataset(tst_sents)
        else:
            self._is_nli = True
            dat = FlairDatapointDataset(list(self.preprocess_train_datapoints(dps)))
            try:
                dev_tps = list(
                    self._compute_hard_labels(  # needed to avoid training error with flair
                        self.compute_hlv(self._read_nli_file(data_dir / "dev.jsonl"))
                    )
                )
            except FileNotFoundError:
                dev_dat = None
            else:
                dev_dat = FlairDatapointDataset(dev_tps)
            try:
                tst_tps = list(
                    self.compute_hlv(self._read_nli_file(data_dir / "test.jsonl"))
                )
            except FileNotFoundError:
                tst_dat = None
            else:
                tst_dat = FlairDatapointDataset(tst_tps)
        super().__init__(dat, dev_dat, tst_dat, sample_missing_splits=False)

    @classmethod
    def _read_nli_file(cls, path: Path, /) -> Iterator["NLIDataPoint"]:
        with path.open(encoding="utf8") as f:
            for line in tqdm(f, desc=f"Reading {path}"):
                obj = json.loads(line)
                dp = cls._input2nlidp(obj["input"])
                dp.add_label(cls._nli_hard_label_type, obj["output"]["verdict"])
                for k, v in obj.get("metadata", {}).items():
                    dp.add_metadata(k, v)
                yield dp

    def _read_tc_file(self, path: Path, /) -> Iterator["TCDataPoint"]:
        with path.open(encoding="utf8") as f:
            for line in tqdm(f, desc=f"Reading {path}"):
                obj = json.loads(line)
                s = TCDataPoint(obj["input"]["text"])
                s.tokens = s.tokens[: self.__max_length]
                output = obj["output"]
                if self._mullab is None:
                    try:
                        label = output["class"]
                    except KeyError:
                        labels = output["classes"]
                        self._mullab = True
                    else:
                        labels = [str(label)]
                        self._mullab = False
                else:
                    labels = (
                        output["classes"] if self._mullab else [str(output["class"])]
                    )
                for l in labels:
                    s.add_label(self._bc_hard_label_type, l)
                for k, v in obj.get("metadata", {}).items():
                    s.add_metadata(k, v)
                yield s

    @overload
    def preprocess_train_datapoints(
        self, dps: Iterable["TCDataPoint"], /
    ) -> Iterator["TCDataPoint"]:
        ...

    @overload
    def preprocess_train_datapoints(
        self, dps: Iterable["NLIDataPoint"], /
    ) -> Iterator["NLIDataPoint"]:
        ...

    @abc.abstractmethod
    def preprocess_train_datapoints(self, dps, /):
        raise NotImplementedError

    def make_classifier(
        self, embeddings: TransformerDocumentEmbeddings, label_dict: Dictionary
    ) -> Classifier[Sentence] | Classifier[TextPair]:
        if self._is_nli:
            return TextPairClassifier(
                embeddings, self.hard_label_type, label_dictionary=label_dict
            )
        return TextClassifier(
            embeddings,
            self.hard_label_type,
            label_dictionary=label_dict,
            multi_label=self._mullab,
        )

    @property
    def soft_label_type(self) -> str:
        label_type = "verdict" if self._is_nli else "class"
        return f"soft {label_type}"

    @property
    def hard_label_type(self) -> str:
        return self._nli_hard_label_type if self._is_nli else self._bc_hard_label_type

    @property
    def is_nli(self) -> bool:
        return self._is_nli

    @property
    def multi_label(self) -> Optional[bool]:
        if self._is_nli:
            return False
        return self._mullab

    @staticmethod
    def _input2nlidp(obj: dict, /) -> "NLIDataPoint":
        try:
            text = obj["premise"]
        except KeyError:
            raise NotNLICorpusError
        return NLIDataPoint(IntactSentence(text), IntactSentence(obj["hypothesis"]))

    @overload
    def compute_hlv(self, dps: Iterable["TCDataPoint"], /) -> Iterator["TCDataPoint"]:
        ...

    @overload
    def compute_hlv(self, dps: Iterable["NLIDataPoint"], /) -> Iterator["NLIDataPoint"]:
        ...

    def compute_hlv(self, dps, /):
        dp2counts = Counter()
        dp2counter = defaultdict(Counter)
        for dp in dps:
            dp2counts[dp] += 1
            for l in self._get_hard_labels(dp):
                dp2counter[dp][_HashableLabel(l)] += 1
        for dp, counter in dp2counter.items():
            dp.remove_labels(self.hard_label_type)
            val2cnt = defaultdict(int, {l.value: c for l, c in counter.items()})
            for val in self.__labels:
                dp.add_label(self.soft_label_type, val, val2cnt[val] / dp2counts[dp])
            yield dp

    @overload
    def _compute_hard_labels(
        self, dps: Iterable["TCDataPoint"], /
    ) -> Iterator["TCDataPoint"]:
        ...

    @overload
    def _compute_hard_labels(
        self, dps: Iterable["NLIDataPoint"], /
    ) -> Iterator["NLIDataPoint"]:
        ...

    def _compute_hard_labels(self, dps, /):
        for dp in dps:
            max_label = max(dp.get_labels(self.soft_label_type), key=lambda l: l.score)
            dp.add_label(self.hard_label_type, max_label.value)
            yield dp

    def _get_hard_labels(self, dp: Union["NLIDataPoint", "TCDataPoint"]) -> list[Label]:
        return (
            dp.get_labels(self.hard_label_type)
            if not self._is_nli and self._mullab
            else [dp.get_label(self.hard_label_type)]
        )


class DisaggregatedTrainSetCorpus(HLVCorpus):
    def preprocess_train_datapoints(self, dps, /):
        yield from dps


class MajorityVotedTrainSetCorpus(HLVCorpus):
    def preprocess_train_datapoints(self, dps, /):
        for dp in self.compute_hlv(dps):
            if self.multi_label:
                for l in dp.get_labels(self.soft_label_type):
                    if l.score > 0.5:
                        dp.add_label(self.hard_label_type, l.value)
            else:
                label2hlv = {
                    l.value: l.score for l in dp.get_labels(self.soft_label_type)
                }
                mv_label = max(label2hlv, key=label2hlv.get)
                dp.add_label(self.hard_label_type, mv_label)
            dp.remove_labels(self.soft_label_type)
            yield dp


class AmbiguouslyLabelledTrainSetCorpus(HLVCorpus):
    """Gajewska's (2023) ambiguous labelling, extended to >2 classes."""

    def preprocess_train_datapoints(self, dps, /):
        if self.multi_label:
            raise NotImplementedError
        for dp in self.compute_hlv(dps):
            for label in dp.get_labels(self.soft_label_type):
                if label.score > 0:
                    new_dp = dp.duplicate()
                    new_dp.add_label(self.hard_label_type, label.value)
                    yield new_dp


class SoftLabelledTrainSetCorpus(HLVCorpus):
    def preprocess_train_datapoints(self, dps, /):
        yield from self.compute_hlv(dps)

    def make_classifier(
        self, embeddings: TransformerDocumentEmbeddings, label_dict: Dictionary
    ) -> TextSoftClassifier | TextPairSoftClassifier:
        cls = TextPairSoftClassifier if self.is_nli else TextSoftClassifier
        return cls(
            embeddings,
            self.soft_label_type,
            label_dictionary=label_dict,
            multi_label=self.multi_label,
        )


class SoftLabelledAndMajorityVotedTrainSetCorpus(HLVCorpus):
    def preprocess_train_datapoints(self, dps, /):
        for dp in self.compute_hlv(dps):
            if self.multi_label:
                for l in dp.get_labels(self.soft_label_type):
                    if l.score > 0.5:
                        dp.add_label(self.hard_label_type, l.value)
            else:
                label2hlv = {
                    l.value: l.score for l in dp.get_labels(self.soft_label_type)
                }
                mv_label = max(label2hlv, key=label2hlv.get)
                dp.add_label(self.hard_label_type, mv_label)
            dp.add_label("multitask_id", self.soft_label_type)
            dp.add_label("multitask_id", self.hard_label_type)
            yield dp

    def make_classifier(
        self, embeddings: TransformerDocumentEmbeddings, label_dict: Dictionary
    ) -> SingleEvaluationTaskMultitaskModel:
        soft_cls = TextPairSoftClassifier if self.is_nli else TextSoftClassifier
        hard_cls = TextPairClassifier if self.is_nli else TextClassifier
        return SingleEvaluationTaskMultitaskModel(
            [
                soft_cls(
                    embeddings,
                    self.soft_label_type,
                    label_dictionary=label_dict,
                    multi_label=self.multi_label,
                ),
                hard_cls(
                    embeddings,
                    self.hard_label_type,
                    label_dictionary=label_dict,
                    multi_label=self.multi_label,
                ),
            ],
            task_ids=[self.soft_label_type, self.hard_label_type],
            use_all_tasks=True,
        )


class AnnotatorLabelledTrainSetCorpus(HLVCorpus):
    def preprocess_train_datapoints(self, dps, /):
        dp2label2anns = defaultdict(lambda: defaultdict(list))
        for dp in dps:
            if dp.has_label(self.hard_label_type):
                for l in dp.get_labels(self.hard_label_type):
                    dp2label2anns[dp][_HashableLabel(l)].append(
                        dp.get_metadata("annotator")
                    )
        ann_dict = Dictionary(add_unk=False)
        for dp, label2anns in dp2label2anns.items():
            dp.remove_labels(self.hard_label_type)
            for label, anns in label2anns.items():
                for ann in anns:
                    ann_dict.add_item(str(ann))
                    dp.add_label(
                        f"{self.hard_label_type} by annotator {ann}", label.value
                    )
            yield dp
        self.__ann_dict = ann_dict

    def make_classifier(
        self, embeddings: TransformerDocumentEmbeddings, label_dict: Dictionary
    ):
        if self.is_nli:
            cls = AnnotatorEnsembleTextPairClassifier
            kwargs = {}
        else:
            cls = AnnotatorEnsembleTextClassifier  # type: ignore[assignment]
            kwargs = {"multi_label": self.multi_label}
        return cls(
            embeddings,
            self.hard_label_type,
            label_dict,
            annotator_label_type_fmt=f"{self.hard_label_type} by annotator {{0}}",
            annotator_dict=self.__ann_dict,
            **kwargs,
        )


class AnnotatorRankingScoredTrainSetCorpus(HLVCorpus):
    def __init__(
        self,
        data_dir: Path,
        labels: Iterable[str],
        sort_train_desc: bool = False,
        soft_scoring: bool = False,
        max_length: Optional[int] = None,
    ) -> None:
        self.soft_scoring = soft_scoring
        self.__labels = labels
        kwargs = {}
        if max_length is not None:
            kwargs["max_length"] = max_length
        super().__init__(data_dir, labels, sort_train_desc, **kwargs)

    def preprocess_train_datapoints(self, dps, /):
        dps = list(dps)
        assert self.multi_label is not None
        if self.multi_label:
            dp2anninfo = defaultdict(list)
            for dp in dps:
                dp2anninfo[dp].append(
                    (
                        dp.get_metadata("annotator"),
                        {l.value for l in dp.get_labels(self.hard_label_type)},
                    )
                )
            # number of dps ann (p)articipates in
            ann2pcounts = Counter()
            # number of dps ann (m)atches the majority on a label
            ann2label2mcounts = defaultdict(Counter)
            ann2label2stotal = defaultdict(lambda: defaultdict(float))
            for dp, anninfo in dp2anninfo.items():
                ann2pcounts.update(a for a, _ in anninfo)
                counter = Counter(l for _, ls in anninfo for l in ls)
                label2prob = {l: counter[l] / len(anninfo) for l in self.__labels}
                assert all(p <= 1 for p in label2prob.values())
                mv_labels = {l for l, p in label2prob.items() if p > 0.5}
                dp.remove_labels(self.hard_label_type)
                for l in mv_labels:
                    dp.add_label(self.hard_label_type, l)
                for ann, labels in anninfo:
                    for lab, prob in label2prob.items():
                        if lab in mv_labels:
                            ann2label2mcounts[ann][lab] += 1 if lab in labels else 0
                            ann2label2stotal[ann][lab] += prob if lab in labels else 0
                        else:
                            ann2label2mcounts[ann][lab] += 1 if lab not in labels else 0
                            ann2label2stotal[ann][lab] += (
                                (1 - prob) if lab not in labels else 0
                            )
            for dp, anninfo in dp2anninfo.items():
                if self.soft_scoring:
                    score = sum(
                        sum(
                            (ann2label2stotal[a][l] / c)
                            if (c := ann2label2mcounts[a][l])
                            else 0
                            for a, _ in anninfo
                        )
                        for l in self.__labels
                    ) / len(self.__labels)
                else:
                    score = sum(
                        sum(
                            ann2label2mcounts[a][l] / ann2pcounts[a] for a, _ in anninfo
                        )
                        for l in self.__labels
                    ) / len(self.__labels)
                dp.add_metadata("weight", score)
                yield dp
        else:
            dp2anninfo = defaultdict(list)
            for dp in dps:
                dp2anninfo[dp].append(
                    (
                        dp.get_metadata("annotator"),
                        dp.get_label(self.hard_label_type).value,
                    )
                )
            ann2pcounts = Counter()  # number of dps ann (p)articipates in
            ann2mcounts = Counter()  # number of dps ann (m)atches the majority
            ann2stotal = defaultdict(float)
            for dp, anninfo in dp2anninfo.items():
                ann2pcounts.update(a for a, _ in anninfo)
                counter = Counter(l for _, l in anninfo)
                mv_label = max(counter, key=counter.get)
                mv_prob = counter[mv_label] / sum(counter.values())
                for a, l in anninfo:
                    if l == mv_label:
                        ann2mcounts[a] += 1
                        ann2stotal[a] += mv_prob
                dp.set_label(self.hard_label_type, mv_label)
            for dp, anninfo in dp2anninfo.items():
                mv_label = dp.get_label(self.hard_label_type).value
                if self.soft_scoring:
                    score = sum(
                        (ann2stotal[a] / ann2mcounts[a]) if ann2mcounts[a] else 0
                        for a, _ in anninfo
                    )
                else:
                    score = sum(ann2mcounts[a] / ann2pcounts[a] for a, _ in anninfo)
                dp.set_label(self.hard_label_type, mv_label, score)
                yield dp

    def make_classifier(
        self, embeddings: TransformerDocumentEmbeddings, label_dict: Dictionary
    ):
        return InstWeightingTextClassifier(
            embeddings,
            self.hard_label_type,
            label_dictionary=label_dict,
            multi_label=self.multi_label,
        )


class AnnotationCollectedTrainSetCorpus(HLVCorpus):
    def preprocess_train_datapoints(self, dps, /):
        dp2annotations = defaultdict(list)
        for dp in dps:
            dp2annotations[dp].append(self._get_hard_labels(dp))
        for dp, annotations in dp2annotations.items():
            dp.remove_labels(self.hard_label_type)
            for i, annotation in enumerate(annotations):
                for label in annotation:
                    dp.add_label(f"annotation {i}", label.value)
            dp.add_metadata("num_annotations", len(annotations))
            yield dp

    def make_classifier(
        self, embeddings: TransformerDocumentEmbeddings, label_dict: Dictionary
    ):
        if self.is_nli:
            cls = LossAggregationTextPairClassifier
        else:
            cls = LossAggregationTextClassifier
        return cls(
            embeddings,
            "@@UNUSED@@",
            label_dictionary=label_dict,
            multi_label=self.multi_label,
        )


class TrainSetOnlyCorpus(DisaggregatedTrainSetCorpus):
    def __init__(self, data_dir: Path) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            (Path(tmpdirname) / "train.jsonl").write_text(
                (data_dir / "train.jsonl").read_text("utf8"), "utf8"
            )
            super().__init__(Path(tmpdirname), [])


class IntactSentence(Sentence):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.add_metadata("orig_text", text)


class TCDataPoint(IntactSentence):
    def to_input_dict(self) -> dict[str, dict]:
        return {"text": self.get_metadata("orig_text")}

    def duplicate(self) -> "TCDataPoint":
        return self.__class__(self.get_metadata("orig_text"))

    def __hash__(self):
        return hash(self.get_metadata("orig_text"))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.get_metadata("orig_text") == other.get_metadata("orig_text")


class NLIDataPoint(DataPair[IntactSentence, IntactSentence]):
    def to_input_dict(self) -> dict[str, dict]:
        return {
            "premise": self.first.get_metadata("orig_text"),
            "hypothesis": self.second.get_metadata("orig_text"),
        }

    # TODO use .get_metadata('orig_text') instead of .text
    def duplicate(self) -> "NLIDataPoint":
        return self.__class__(
            IntactSentence(self.first.text), IntactSentence(self.second.text)
        )

    # TODO use .get_metadata('orig_text') instead of .text
    def __hash__(self):
        return hash((self.first.text, self.second.text))

    # TODO use .get_metadata('orig_text') instead of .text
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.first.text == other.first.text
            and self.second.text == other.second.text
        )


class _HashableLabel:
    def __init__(self, label: Label, /) -> None:
        self._label = label

    @property
    def value(self):
        return self._label.value

    @property
    def score(self):
        return self._label.score

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.value == other.value
