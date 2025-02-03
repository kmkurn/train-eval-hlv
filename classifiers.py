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

from typing import Optional, cast

import flair
import torch
from flair.data import Dictionary, Sentence, TextPair
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import MultitaskModel, TextClassifier, TextPairClassifier
from flair.nn import Classifier


class TextSoftClassifier(TextClassifier):
    def _prepare_label_tensor(
        self, prediction_data_points: list[Sentence]
    ) -> torch.Tensor:
        dps = prediction_data_points
        label_tensor = torch.empty(
            [len(dps), len(self.label_dictionary)], device=flair.device
        )
        for i, dp in enumerate(dps):
            for label in dp.get_labels(self.label_type):
                j = self.label_dictionary.get_idx_for_item(label.value)
                label_tensor[i, j] = label.score
        return label_tensor

    def predict(
        self,
        sentences: list[Sentence] | Sentence,
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = True,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: str = "none",
    ) -> None:
        return super().predict(
            sentences,
            mini_batch_size,
            return_probabilities_for_all_classes=True,
            verbose=verbose,
            label_name=label_name,
            return_loss=return_loss,
            embedding_storage_mode=embedding_storage_mode,
        )


class TextPairSoftClassifier(TextPairClassifier):
    def _prepare_label_tensor(
        self, prediction_data_points: list[TextPair]
    ) -> torch.Tensor:
        dps = prediction_data_points
        label_tensor = torch.empty(
            [len(dps), len(self.label_dictionary)], device=flair.device
        )
        for i, dp in enumerate(dps):
            for label in dp.get_labels(self.label_type):
                j = self.label_dictionary.get_idx_for_item(label.value)
                label_tensor[i, j] = label.score
        return label_tensor

    def predict(
        self,
        textpairs: list[TextPair] | TextPair,
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = True,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: str = "none",
    ) -> None:
        return super().predict(
            textpairs,
            mini_batch_size,
            return_probabilities_for_all_classes=True,
            verbose=verbose,
            label_name=label_name,
            return_loss=return_loss,
            embedding_storage_mode=embedding_storage_mode,
        )


class SingleEvaluationTaskMultitaskModel(MultitaskModel):
    def evaluate(
        self,
        data_points,
        gold_label_type,
        out_path=None,
        main_evaluation_metric=("micro avg", "f1-score"),
        evaluate_all=False,
        **kwargs,
    ):
        return super().evaluate(
            data_points,
            gold_label_type,
            out_path,
            main_evaluation_metric,
            evaluate_all=False,
            **kwargs,
        )


class AnnotatorEnsembleTextClassifier(Classifier[Sentence]):
    def __init__(
        self,
        embeddings: TransformerDocumentEmbeddings,
        label_type: str,
        label_dict: Dictionary,
        annotator_label_type_fmt: str,
        annotator_dict: Dictionary,
        multi_label: bool = False,
    ) -> None:
        super().__init__()
        self._embeddings = embeddings
        self._label_type = label_type
        self._label_dict = label_dict
        self._ann_label_type_fmt = annotator_label_type_fmt
        self._ann_dict = annotator_dict
        self.decoder = MultiheadLinear(
            len(annotator_dict), embeddings.embedding_length, len(label_dict)
        )
        self.loss_fn: torch.nn.Module
        if multi_label:
            self.loss_fn = SumBCEWithLogitsLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.averages_probs = True
        self.multi_label = multi_label
        self.to(flair.device)

    def _get_state_dict(self):
        # Adapted from flair's TextClassifier
        return {
            **super()._get_state_dict(),
            "document_embeddings": self._embeddings.save_embeddings(
                use_state_dict=False
            ),
            "label_type": self._label_type,
            "label_dict": self._label_dict,
            "annotator_label_type_fmt": self._ann_label_type_fmt,
            "annotator_dict": self._ann_dict,
            "multi_label": self.multi_label,
            "averages_probs": self.averages_probs,
        }

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        # Adapted from flair's TextClassifier
        avg_probs = state.pop("averages_probs")
        model = super()._init_model_with_state_dict(
            state,
            embeddings=state.pop("document_embeddings"),
            label_type=state.pop("label_type"),
            label_dict=state.pop("label_dict"),
            annotator_label_type_fmt=state.pop("annotator_label_type_fmt"),
            annotator_dict=state.pop("annotator_dict"),
            multi_label=state.pop("multi_label"),
        )
        model.averages_probs = avg_probs
        return model

    @property
    def label_type(self) -> str:
        return self._label_type

    def forward_loss(self, sentences: list[Sentence]) -> tuple[torch.Tensor, int]:
        x = self._embed_sentences(sentences)
        assert x.shape == (len(sentences), self._embeddings.embedding_length)
        x = self.decoder(x)
        assert x.shape == (len(self._ann_dict), len(sentences), len(self._label_dict))
        if self.multi_label:
            y = torch.stack(
                [
                    self._make_multilabel_target_tensor(
                        *self._get_sentence_multilabels(s)
                    )
                    for s in sentences
                ]
            )
            assert y.shape == (
                len(sentences),
                len(self._label_dict),
                len(self._ann_dict),
            )
        else:
            y = torch.stack(
                [
                    self._make_target_tensor(*self._get_sentence_labels(s))
                    for s in sentences
                ]
            )
            assert y.shape == (len(sentences), len(self._ann_dict))
        x = x.transpose(0, 1).transpose(1, 2)
        assert x.shape == (len(sentences), len(self._label_dict), len(self._ann_dict))
        count = (y != self.loss_fn.ignore_index).long().sum().item()
        return self.loss_fn(x, y), cast(int, count)

    def predict(
        self,
        sentences: list[Sentence] | Sentence,
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: str = "none",
    ):
        if verbose or embedding_storage_mode != "none":
            raise NotImplementedError
        if label_name is None:
            label_name = self.label_type

        overall_loss = torch.tensor(0.0, device=flair.device)
        overall_count = 0
        for batch in DataLoader(FlairDatapointDataset(sentences), mini_batch_size):
            with torch.no_grad():
                x = self._embed_sentences(batch)
                assert x.shape == (len(batch), self._embeddings.embedding_length)
                x = self.decoder(x)
                assert x.shape == (
                    len(self._ann_dict),
                    len(batch),
                    len(self._label_dict),
                )

                if return_loss:
                    if self.multi_label:
                        y = torch.stack(
                            [
                                self._make_multilabel_target_tensor(
                                    *self._get_sentence_multilabels(s)
                                )
                                for s in sentences
                            ]
                        )
                        assert y.shape == (
                            len(batch),
                            len(self._label_dict),
                            len(self._ann_dict),
                        )
                    else:
                        y = torch.stack(
                            [
                                self._make_target_tensor(*self._get_sentence_labels(s))
                                for s in batch
                            ]
                        )
                        assert y.shape == (len(batch), len(self._ann_dict))
                    x = x.transpose(0, 1).transpose(1, 2)
                    assert x.shape == (
                        len(batch),
                        len(self._label_dict),
                        len(self._ann_dict),
                    )
                    overall_loss += self.loss_fn(x, y)
                    overall_count += cast(
                        int, (y != self.loss_fn.ignore_index).long().sum().item()
                    )
                    x = x.transpose(1, 2).transpose(0, 1)
                    assert x.shape == (
                        len(self._ann_dict),
                        len(batch),
                        len(self._label_dict),
                    )

                if self.multi_label:
                    probs = torch.nn.functional.sigmoid(x)
                else:
                    probs = torch.nn.functional.softmax(x, dim=-1)
                if self.averages_probs:
                    probs = probs.mean(dim=0)
                elif self.multi_label:
                    probs = torch.where(
                        probs > 0.5, torch.ones_like(probs), torch.zeros_like(probs)
                    ).mean(dim=0)
                else:
                    _, preds = probs.max(dim=-1)
                    assert preds.shape == (len(self._ann_dict), len(batch))
                    probs = (
                        torch.zeros_like(probs)
                        .scatter(dim=-1, index=preds.unsqueeze(-1), value=1.0)
                        .mean(dim=0)
                    )
                assert probs.shape == (len(batch), len(self._label_dict))

                if return_probabilities_for_all_classes:
                    for i, s in enumerate(batch):
                        for j in range(len(self._label_dict)):
                            s.add_label(
                                label_name,
                                self._label_dict.get_item_for_index(j),
                                score=probs[i, j].item(),
                            )
                elif self.multi_label:
                    for i, s in enumerate(batch):
                        for j in range(len(self._label_dict)):
                            score = probs[i, j].item()
                            if score > 0.5:
                                s.add_label(
                                    label_name,
                                    self._label_dict.get_item_for_index(j),
                                    score,
                                )
                else:
                    max_probs, preds = probs.max(dim=-1)
                    assert max_probs.shape == (len(batch),)
                    assert max_probs.shape == preds.shape
                    for i, s in enumerate(batch):
                        s.add_label(
                            label_name,
                            self._label_dict.get_item_for_index(preds[i]),
                            score=max_probs[i].item(),
                        )

        if return_loss:
            return overall_loss, overall_count

    def _embed_sentences(self, sentences: list[Sentence], /) -> torch.Tensor:
        self._embeddings.embed(sentences)
        return torch.stack(
            [s.get_embedding(self._embeddings.get_names()) for s in sentences]
        )

    def _get_sentence_labels(
        self, sentence: Sentence, /
    ) -> tuple[list[str], list[str]]:
        labs, anns = [], []
        for ann in self._ann_dict.get_items():
            labtype = self._ann_label_type_fmt.format(ann)
            if sentence.has_label(labtype):
                labs.append(sentence.get_label(labtype).value)
                anns.append(ann)
        return labs, anns

    def _get_sentence_multilabels(
        self, sentence: Sentence, /
    ) -> tuple[list[list[str]], list[str]]:
        annotations, annotators = [], []
        for ann in self._ann_dict.get_items():
            labtype = self._ann_label_type_fmt.format(ann)
            if sentence.has_label(labtype):
                annotations.append([l.value for l in sentence.get_labels(labtype)])
                annotators.append(ann)
        return annotations, annotators

    def _make_target_tensor(self, labs: list[str], anns: list[str]) -> torch.Tensor:
        t = torch.full(
            [len(self._ann_dict)], self.loss_fn.ignore_index, device=flair.device
        )
        for ann, lab in zip(anns, labs):
            a = self._ann_dict.get_idx_for_item(ann)
            l = self._label_dict.get_idx_for_item(lab)
            t[a] = l
        return t

    def _make_multilabel_target_tensor(
        self, annotations: list[list[str]], annotators: list[str]
    ) -> torch.Tensor:
        t = torch.full(
            [len(self._label_dict), len(self._ann_dict)],
            self.loss_fn.ignore_index,
            device=flair.device,
        )
        for ann, labels in zip(annotators, annotations):
            a = self._ann_dict.get_idx_for_item(ann)
            t[:, a] = 0
            for l in self._label_dict.get_idx_for_items(labels):
                t[l, a] = 1
        return t


class AnnotatorEnsembleTextPairClassifier(Classifier[TextPair]):
    def __init__(
        self,
        embeddings: TransformerDocumentEmbeddings,
        label_type: str,
        label_dict: Dictionary,
        annotator_label_type_fmt: str,
        annotator_dict: Dictionary,
    ) -> None:
        super().__init__()
        self._embeddings = embeddings
        self._label_type = label_type
        self._label_dict = label_dict
        self._ann_label_type_fmt = annotator_label_type_fmt
        self._ann_dict = annotator_dict
        self.decoder = MultiheadLinear(
            len(annotator_dict), embeddings.embedding_length, len(label_dict)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        self.averages_probs = True
        self.to(flair.device)

    @property
    def label_type(self) -> str:
        return self._label_type

    def forward_loss(self, textpairs: list[TextPair]) -> tuple[torch.Tensor, int]:
        x = self._embed_textpairs(textpairs)
        assert x.shape == (len(textpairs), self._embeddings.embedding_length)
        x = self.decoder(x)
        assert x.shape == (len(self._ann_dict), len(textpairs), len(self._label_dict))
        y = torch.stack(
            [
                self._make_target_tensor(*self._get_textpair_labels(tp))
                for tp in textpairs
            ]
        )
        assert y.shape == (len(textpairs), len(self._ann_dict))
        x = x.transpose(0, 1).transpose(1, 2)
        assert x.shape == (len(textpairs), len(self._label_dict), len(self._ann_dict))
        count = (y != self.loss_fn.ignore_index).long().sum().item()
        return self.loss_fn(x, y), cast(int, count)

    def predict(
        self,
        textpairs: list[TextPair] | TextPair,
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss: bool = False,
        embedding_storage_mode: str = "none",
    ):
        if verbose or embedding_storage_mode != "none":
            raise NotImplementedError
        if label_name is None:
            label_name = self.label_type

        overall_loss = torch.tensor(0.0, device=flair.device)
        overall_count = 0
        for batch in DataLoader(FlairDatapointDataset(textpairs), mini_batch_size):
            with torch.no_grad():
                x = self._embed_textpairs(batch)
                assert x.shape == (len(batch), self._embeddings.embedding_length)
                x = self.decoder(x)
                assert x.shape == (
                    len(self._ann_dict),
                    len(batch),
                    len(self._label_dict),
                )

                if return_loss:
                    y = torch.stack(
                        [
                            self._make_target_tensor(*self._get_textpair_labels(tp))
                            for tp in batch
                        ]
                    )
                    assert y.shape == (len(batch), len(self._ann_dict))
                    x = x.transpose(0, 1).transpose(1, 2)
                    assert x.shape == (
                        len(batch),
                        len(self._label_dict),
                        len(self._ann_dict),
                    )
                    overall_loss += self.loss_fn(x, y)
                    overall_count += cast(
                        int, (y != self.loss_fn.ignore_index).long().sum().item()
                    )
                    x = x.transpose(1, 2).transpose(0, 1)
                    assert x.shape == (
                        len(self._ann_dict),
                        len(batch),
                        len(self._label_dict),
                    )

                probs = torch.nn.functional.softmax(x, dim=-1)
                if self.averages_probs:
                    probs = probs.mean(dim=0)
                else:
                    _, preds = probs.max(dim=-1)
                    assert preds.shape == (len(self._ann_dict), len(batch))
                    probs = (
                        torch.zeros_like(probs)
                        .scatter(dim=-1, index=preds.unsqueeze(-1), value=1.0)
                        .mean(dim=0)
                    )
                assert probs.shape == (len(batch), len(self._label_dict))

                if return_probabilities_for_all_classes:
                    for i, tp in enumerate(batch):
                        for j in range(len(self._label_dict)):
                            tp.add_label(
                                label_name,
                                self._label_dict.get_item_for_index(j),
                                score=probs[i, j].item(),
                            )
                else:
                    max_probs, preds = probs.max(dim=-1)
                    assert max_probs.shape == (len(batch),)
                    assert max_probs.shape == preds.shape
                    for i, tp in enumerate(batch):
                        tp.add_label(
                            label_name,
                            self._label_dict.get_item_for_index(preds[i]),
                            score=max_probs[i].item(),
                        )

        if return_loss:
            return overall_loss, overall_count

    def _embed_textpairs(self, textpairs: list[TextPair], /) -> torch.Tensor:
        assert (
            self._embeddings.tokenizer.sep_token is not None
        ), "embeddings must support [SEP] token"
        sep = self._embeddings.tokenizer.sep_token
        sents = []
        for tp in textpairs:
            first = tp.first.to_tokenized_string()
            second = tp.second.to_tokenized_string()
            sents.append(Sentence(f"{first} {sep} {second}", use_tokenizer=False))
        self._embeddings.embed(sents)
        return torch.stack(
            [s.get_embedding(self._embeddings.get_names()) for s in sents]
        )

    def _get_textpair_labels(
        self, textpair: TextPair, /
    ) -> tuple[list[str], list[str]]:
        labs, anns = [], []
        for ann in self._ann_dict.get_items():
            labtype = self._ann_label_type_fmt.format(ann)
            if textpair.has_label(labtype):
                labs.append(textpair.get_label(labtype).value)
                anns.append(ann)
        return labs, anns

    def _make_target_tensor(self, labs: list[str], anns: list[str]) -> torch.Tensor:
        t = torch.full(
            [len(self._ann_dict)], self.loss_fn.ignore_index, device=flair.device
        )
        for ann, lab in zip(anns, labs):
            a = self._ann_dict.get_idx_for_item(ann)
            l = self._label_dict.get_idx_for_item(lab)
            t[a] = l
        return t


class InstWeightingTextClassifier(TextClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function.reduction = "none"

    def forward_loss(self, sentences: list[Sentence]) -> tuple[torch.Tensor, int]:
        losses, _ = super().forward_loss(sentences)
        if self.multi_label:
            assert losses.shape == (len(sentences), len(self.label_dictionary))
            losses = losses.transpose(0, 1)
            weights = torch.tensor(
                [s.get_metadata("weight") for s in sentences], device=flair.device
            )
        else:
            assert losses.shape == (len(sentences),)
            weights = torch.tensor(
                [s.get_label(self.label_type).score for s in sentences],
                device=flair.device,
            )
        loss = (losses * weights).sum()
        return loss, losses.numel()

    def evaluate(self, *args, **kwargs):
        # Avoid error shape mismatch when computing loss on dev set
        kwargs["return_loss"] = False
        return super().evaluate(*args, **kwargs)


class LossAggregationTextClassifier(TextClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.multi_label:
            loss_cls = AggOverAnnsBCEWithLogitsLoss
        else:
            loss_cls = AggOverAnnsCrossEntropyLoss
        self.loss_function = loss_cls()

    def _prepare_label_tensor(
        self, prediction_data_points: list[Sentence]
    ) -> torch.Tensor:
        max_annotations = max(
            s.get_metadata("num_annotations") for s in prediction_data_points
        )
        if self.multi_label:
            label_tensor = torch.full(
                [
                    len(prediction_data_points),
                    max_annotations,
                    len(self.label_dictionary),
                ],
                self.loss_function.ignore_index,
                device=flair.device,
            )
            for s_id, sent in enumerate(prediction_data_points):
                for a_id in range(sent.get_metadata("num_annotations")):
                    labels = [l.value for l in sent.get_labels(f"annotation {a_id}")]
                    for lab in self.label_dictionary.get_items():
                        l_id = self.label_dictionary.get_idx_for_item(lab)
                        label_tensor[s_id, a_id, l_id] = 1 if lab in labels else 0
        else:
            label_tensor = torch.full(
                [len(prediction_data_points), max_annotations],
                self.loss_function.ignore_index,
                device=flair.device,
            )
            for s_id, sent in enumerate(prediction_data_points):
                for a_id in range(sent.get_metadata("num_annotations")):
                    lab = sent.get_label(f"annotation {a_id}")
                    l_id = self.label_dictionary.get_idx_for_item(lab.value)  # type: ignore[attr-defined]
                    label_tensor[s_id, a_id] = l_id
        return label_tensor


class LossAggregationTextPairClassifier(TextPairClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = AggOverAnnsCrossEntropyLoss()

    def _prepare_label_tensor(
        self, prediction_data_points: list[TextPair]
    ) -> torch.Tensor:
        max_annotations = max(
            s.get_metadata("num_annotations") for s in prediction_data_points
        )
        label_tensor = torch.full(
            [len(prediction_data_points), max_annotations],
            self.loss_function.ignore_index,
            device=flair.device,
        )
        for s_id, sent in enumerate(prediction_data_points):
            for a_id in range(sent.get_metadata("num_annotations")):
                lab = sent.get_label(f"annotation {a_id}")
                l_id = self.label_dictionary.get_idx_for_item(lab.value)
                label_tensor[s_id, a_id] = l_id
        return label_tensor


class MultiheadLinear(torch.nn.Module):
    def __init__(self, n_heads: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty([n_heads, in_features, out_features])
        )
        self.bias = torch.nn.Parameter(torch.empty([n_heads, out_features]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        return x @ self.weight + self.bias.unsqueeze(-2)

    def extra_repr(self) -> str:
        return "n_heads={}, in_features={}, out_features={}".format(*self.weight.shape)


class SumBCEWithLogitsLoss(torch.nn.Module):
    ignore_index = -100

    def __init__(self):
        super().__init__()
        self.__torch_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.__torch_loss_fn(logits, targets.float())
        assert losses.shape == targets.shape
        return losses.masked_select(targets != self.ignore_index).sum()


class AggOverAnnsBCEWithLogitsLoss(torch.nn.Module):
    ignore_index = -100

    def __init__(self) -> None:
        super().__init__()
        self.aggfunc = "min"
        self.__torch_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.__torch_loss_fn(
            logits.unsqueeze(dim=1).expand_as(targets), targets.float()
        )
        assert losses.shape == targets.shape
        is_pad = (targets == self.ignore_index).any(dim=-1)
        return self._agg(losses.sum(dim=-1), is_pad).sum(dim=-1)

    def extra_repr(self) -> str:
        return f"aggfunc={self.aggfunc}"

    def _agg(self, x: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        assert x.shape == is_pad.shape
        assert x.dim() == 2
        if self.aggfunc == "min":
            x.masked_fill_(is_pad, float("inf"))
            return x.min(dim=-1)[0]
        if self.aggfunc == "max":
            x.masked_fill_(is_pad, -float("inf"))
            return x.max(dim=-1)[0]
        if self.aggfunc == "mean":
            x.masked_fill_(is_pad, 0)
            return x.sum(dim=-1) / (~is_pad).float().sum(dim=-1)
        raise ValueError(f"unrecognised aggfunc: {self.aggfunc}")


class AggOverAnnsCrossEntropyLoss(torch.nn.Module):
    ignore_index = -100

    def __init__(self) -> None:
        super().__init__()
        self.aggfunc = "min"
        self.__torch_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_anns = targets.size(1)
        losses = self.__torch_loss_fn(
            logits.unsqueeze(dim=2).expand([-1, -1, num_anns]), targets
        )
        assert losses.shape == targets.shape
        is_pad = targets == self.ignore_index
        return self._agg(losses, is_pad).sum(dim=-1)

    def extra_repr(self) -> str:
        return f"aggfunc={self.aggfunc}"

    def _agg(self, x: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        assert x.shape == is_pad.shape
        assert x.dim() == 2
        if self.aggfunc == "min":
            x.masked_fill_(is_pad, float("inf"))
            return x.min(dim=-1)[0]
        if self.aggfunc == "max":
            x.masked_fill_(is_pad, -float("inf"))
            return x.max(dim=-1)[0]
        if self.aggfunc == "mean":
            x.masked_fill_(is_pad, 0)
            return x.sum(dim=-1) / (~is_pad).float().sum(dim=-1)
        raise ValueError(f"unrecognised aggfunc: {self.aggfunc}")
