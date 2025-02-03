#!/usr/bin/env python

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
import os
from pathlib import Path

import numpy as np
import torch
from flair.data import Dictionary
from flair.embeddings import TransformerDocumentEmbeddings
from flair.embeddings.base import register_embeddings
from flair.trainers import ModelTrainer
from flair.trainers.plugins import MetricRecord, TrainerPlugin
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds

from corpus import AmbiguouslyLabelledTrainSetCorpus as ALCorpus
from corpus import AnnotatorLabelledTrainSetCorpus as AnnLCorpus
from corpus import AnnotatorRankingScoredTrainSetCorpus as ARCorpus
from corpus import DisaggregatedTrainSetCorpus as DCorpus
from corpus import MajorityVotedTrainSetCorpus as MVCorpus
from corpus import SoftLabelledAndMajorityVotedTrainSetCorpus as SLMVCorpus
from corpus import SoftLabelledTrainSetCorpus as SLCorpus
from corpus import AnnotationCollectedTrainSetCorpus as ACCorpus
from corpus import TrainSetOnlyCorpus as TrainCorpus
from evaluation import entropy_correlation as ent_corr
from evaluation import hard_accuracy as hard_acc
from evaluation import hard_micro_f1
from evaluation import hard_precision_recall_fscore_support as hard_prfs
from evaluation import poJSD
from evaluation import soft_accuracy as soft_acc
from evaluation import soft_micro_f1
from evaluation import soft_precision_recall_fscore_support as soft_prfs
from hlv_loss import JSDLoss, MultilabelJSDLoss, SoftMacroF1Loss, SoftMicroF1Loss

ex = Experiment("hlv-metrics")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore[assignment]
if "SACRED_MONGO_URL" in os.environ:
    ex.observers.append(
        MongoObserver(
            url=os.environ["SACRED_MONGO_URL"],
            db_name=os.getenv("SACRED_DB_NAME", "sacred"),
        )
    )


@ex.config
def default():
    # data directory
    data_dir = "data"
    # store training results in this directory
    artifacts_dir = "artifacts"
    # training method [ReL, MV, AL, SL, SLMV, AE, AEh, AR, ARh, SmF1, SmF1-v2, SMF1, JSD, LA-min, LA-max, LA-mean]
    method = "ReL"
    # whether Flair's trainer should skip evaluation on eval sets
    hide_eval_sets_from_trainer = True
    # label inventory (optional)
    labels = []
    # pretrained model name (see https://huggingface.co/models)
    model_name = "roberta-base"
    # batch size
    batch_size = 32
    # chunk batches into this size for grad accumulation
    batch_chunk_size = None
    # learning rate
    lr = 5e-5
    # max epochs
    max_epochs = 10
    # log to Sacred every this number of batches
    log_every = 10
    # whether to sort training data descending by length
    sort_train_desc = False
    # whether to save the trained model
    save_model = True
    # whether to freeze the pretrained model
    freeze_embedding = False
    # whether to attach OOM observer for debugging
    attach_oom_observer = False
    # batch size at test time (i.e. prediction)
    test_batch_size = 32
    # whether to use LoRA for finetuning
    use_lora = False
    # modules to apply LoRA to (ignored if use_lora=False)
    lora_target = None
    # whether to save only the LoRA parameters (ignored if use_lora=False)
    save_only_lora_parameters = True
    # truncate input longer than this (only for text classification)
    max_length = None
    # whether to evaluate on dev set on every epoch
    eval_on_epoch_finished = False


@ex.named_config
def testrun():
    """Single epoch from long to short inputs"""
    max_epochs = 1
    sort_train_desc = True


@ex.named_config
def chaosnli():
    """ChaosNLI dataset"""
    labels = "c n e".split()


@ex.named_config
def chaosnli_Llama_3_8B():
    """ChaosNLI dataset with 8B LLaMA 3"""
    labels = "c n e".split()
    model_name = "meta-llama/Meta-Llama-3-8B"
    use_lora = True


@ex.named_config
def lewidi():
    """Datasets from LeWiDi shared task"""
    labels = "False True".split()
    model_name = "Twitter/twhin-bert-base"


@ex.named_config
def lewidi_Llama_3_8B():
    """Datasets from LeWiDi shared task with 8B LLaMA 3"""
    labels = "False True".split()
    model_name = "meta-llama/Meta-Llama-3-8B"
    use_lora = True
    batch_chunk_size = 8


@ex.named_config
def mfrc_Llama_3_8B():
    """MFRC dataset with 8B LLaMA 3"""
    model_name = "meta-llama/Meta-Llama-3-8B"
    use_lora = True
    batch_chunk_size = 4
    test_batch_size = 4
    max_length = 512


@ex.named_config
def TAG_Llama_3_8B():
    """TAG dataset with 8B LLaMA 3"""
    model_name = "meta-llama/Meta-Llama-3-8B"
    use_lora = True
    lora_target = "all-linear"
    batch_chunk_size = 2
    test_batch_size = 4
    max_length = 512


@ex.capture
def do_evaluation(
    model,
    label_dict,
    datapoints,
    gold_label_type,
    artifacts_dir,
    batch_size=32,
    multilabel=False,
    is_test_set=True,
    _run=None,
    _log=None,
):
    hard_eval = hard_micro_f1 if multilabel else hard_acc
    pred_label_type = f"pred {gold_label_type}"
    model.eval()  # ensure correctness primarily for EvalOnDevPlugin
    model.predict(
        datapoints,
        mini_batch_size=batch_size,
        return_probabilities_for_all_classes=True,
        label_name=pred_label_type,
    )
    # Compute hard eval metric
    y_true, y_pred = np.zeros([2, len(datapoints), len(label_dict)], dtype=float)
    for i, dp in enumerate(datapoints):
        for label in dp.get_labels(gold_label_type):
            y_true[i, label_dict.get_idx_for_item(label.value)] = label.score
        for label in dp.get_labels(pred_label_type):
            y_pred[i, label_dict.get_idx_for_item(label.value)] = label.score
    res = hard_eval(y_true, y_pred)
    retval = res
    # Save predicted HLV
    with (artifacts_dir / f"{'' if is_test_set else 'dev-'}inputs.jsonl").open(
        "w", encoding="utf8"
    ) as f:
        for i, dp in enumerate(datapoints):
            print(
                json.dumps(dp.to_input_dict()),
                file=f,
            )
    np.save(
        artifacts_dir / f"{'' if is_test_set else 'dev-'}hlv.npy",
        np.stack([y_true, y_pred], axis=0),
    )
    # Compute soft overall metrics
    split = "test" if is_test_set else "dev"
    if not (_run is None and _log is None):
        _record_metric(
            f"{split}/hard_{'micro_f1' if multilabel else 'acc'}", res, _run=_run
        )
        names = ["poJSD", "ent_corr", f"soft_{'micro_f1' if multilabel else 'acc'}"]
        fns = [
            lambda x, y: poJSD(x, y, multilabel=multilabel),
            lambda x, y: ent_corr(x, y, multilabel=multilabel),
            soft_micro_f1 if multilabel else soft_acc,
        ]
        for name, fn in zip(names, fns):
            _record_metric(f"{split}/{name}", fn(y_true, y_pred), _run=_run)
        # Compute soft class-wise metrics
        for t, eval_fn in zip(
            "hard soft".split(),
            [lambda x, y: hard_prfs(x, y, multilabel=multilabel), soft_prfs],
        ):
            prfs_res = eval_fn(y_true, y_pred)
            for i, (xs, n) in enumerate(zip(prfs_res, "p r f1 supp".split())):
                # Compute macro-average metrics
                if n != "supp":
                    _record_metric(f"{split}/macro/{t}_{n}", xs.mean(), _run=_run)
                if is_test_set:
                    for l in label_dict.get_items():
                        res = xs[label_dict.get_idx_for_item(l)]
                        fmt = "%.3f"
                        if n == "supp":
                            fmt = "%d" if t == "hard" else "%.2f"
                        _record_metric(f"{split}/{t}_{n}/{l}", res, fmt, _run)

    return retval


class SacredLogMetricsPlugin(TrainerPlugin):
    def __init__(self, run: Run, log_every: int = 10) -> None:
        super().__init__()
        self.__run = run
        self.__log_every = log_every

    @TrainerPlugin.hook
    def metric_recorded(self, record: MetricRecord) -> None:
        if self.__should_log(record):
            try:
                value = record.value.item()
            except AttributeError:
                value = record.value
            self.__run.log_scalar(str(record.name), value, record.global_step)

    def __should_log(self, record: MetricRecord) -> bool:
        is_batch_metric = len(record.name.parts) >= 2 and record.name.parts[1] in (
            "batch_loss",
            "gradient_norm",
        )
        return record.is_scalar and (
            not is_batch_metric or record.global_step % self.__log_every == 0
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__} | log_every: {self.__log_every}"


class EvaluateOnDevPlugin(TrainerPlugin):
    def __init__(
        self,
        dev_datapoints,
        label_dict,
        artifacts_dir,
        batch_size=32,
        logger=None,
        sacred_run=None,
    ):
        super().__init__()
        self.dev_datapoints = dev_datapoints
        self.label_dict = label_dict
        self.batch_size = batch_size
        self.artifacts_dir = artifacts_dir
        self.logger = logger
        self.sacred_run = sacred_run
        self.retval = None

    @TrainerPlugin.hook
    def after_training_epoch(self, epoch):
        dev_datapoints = self.dev_datapoints
        corpus = self.corpus
        model = self.model
        batch_size = self.batch_size
        label_dict = self.label_dict
        artifacts_dir = self.artifacts_dir
        log = self.logger
        run = self.sacred_run

        if log is not None:
            log.info("Evaluating on dev")
        self.retval = do_evaluation(
            model,
            label_dict,
            dev_datapoints,
            corpus.soft_label_type,
            artifacts_dir,
            batch_size,
            corpus.multi_label,
            is_test_set=False,
            _run=run,
            _log=log,
        )


@ex.capture
def _record_metric(name, value, fmt="%.3f", _run=None, _log=None):
    if _log is not None:
        _log.info(f"%s: {fmt}", name, value)
    if _run is not None:
        _run.log_scalar(name, value)


@register_embeddings
class TransformerDocumentEmbeddingsWithLoRA(TransformerDocumentEmbeddings):
    def __init__(
        self,
        lora_config: LoraConfig | dict,
        save_only_lora_parameters=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(lora_config, dict):
            lora_config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, lora_config)
        self._lora_config = lora_config
        self._save_only_lora = save_only_lora_parameters
        if save_only_lora_parameters:
            self._register_state_dict_hook(self.__replace_model_state_dict)
            self._register_load_state_dict_pre_hook(self.__set_model_state_dict)
            self.register_load_state_dict_post_hook(self.__process_incompatible_keys)

    def to_params(self):
        params = super().to_params()
        return {
            **params,
            "lora_config": self._lora_config.to_dict(),
            "save_only_lora_parameters": self._save_only_lora,
        }

    def __replace_model_state_dict(self, module, state_dict, prefix, *args):
        assert self is module
        for name in list(state_dict.keys()):
            if name.startswith(f"{prefix}model."):
                state_dict.pop(name)
        for name, value in get_peft_model_state_dict(self.model).items():
            state_dict[f"{prefix}model.{name}"] = value

    def __set_model_state_dict(self, state_dict, prefix, *args):
        set_peft_model_state_dict(
            self.model,
            {
                name[len(f"{prefix}model.") :]: param
                for name, param in state_dict.items()
                if name.startswith(f"{prefix}model.")
            },
        )
        self.__prefix = prefix

    def __process_incompatible_keys(self, module, incompatible_keys):
        for key in list(incompatible_keys.missing_keys):
            if key.startswith(f"{self.__prefix}model."):
                incompatible_keys.missing_keys.remove(key)
        for key in list(incompatible_keys.unexpected_keys):
            if key.startswith(f"{self.__prefix}model."):
                incompatible_keys.unexpected_keys.remove(key)


@ex.automain
def train(
    data_dir,
    artifacts_dir,
    method="ReL",
    hide_eval_sets_from_trainer=False,
    use_lora=False,
    lora_target=None,
    labels=None,
    model_name="bert-base-uncased",
    batch_size=32,
    max_epochs=10,
    log_every=10,
    sort_train_desc=False,
    lr=5e-5,
    save_model=True,
    freeze_embedding=False,
    attach_oom_observer=False,
    test_batch_size=32,
    save_only_lora_parameters=True,
    batch_chunk_size=None,
    max_length=None,
    eval_on_epoch_finished=False,
    _log=None,
    _run=None,
):
    """Train a model."""
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)

    artifacts_dir.mkdir()

    # Compute and save label inventory
    if not labels:
        corpus = TrainCorpus(data_dir)
        label_dict = corpus.make_label_dictionary(corpus.hard_label_type)
    else:
        label_dict = Dictionary(add_unk=False)
        for l in labels:
            label_dict.add_item(l)
    label_dict.save(artifacts_dir / "label.dict")

    kwargs = {}
    if method == "ReL":
        corpus_cls = DCorpus
    elif method == "MV":
        corpus_cls = MVCorpus
    elif method == "AL":
        corpus_cls = ALCorpus
    elif method == "SL":
        corpus_cls = SLCorpus
    elif method == "SLMV":
        corpus_cls = SLMVCorpus
    elif method in ("AE", "AEh"):
        corpus_cls = AnnLCorpus
    elif method in ("AR", "ARh"):
        corpus_cls = ARCorpus
        kwargs.update({"soft_scoring": method == "AR"})
    elif method in ("SmF1", "SmF1-v2", "SMF1"):
        corpus_cls = SLCorpus
    elif method == "JSD":
        corpus_cls = SLCorpus
    elif method.startswith("LA-"):
        corpus_cls = ACCorpus
    else:
        raise ValueError(f"method={method} isn't recognised")

    if max_length:
        kwargs["max_length"] = max_length

    corpus = corpus_cls(data_dir, label_dict.get_items(), sort_train_desc, **kwargs)
    if _log is not None:
        _log.info(
            "Corpus is detected as %s",
            "multilabel" if corpus.multi_label else "multiclass",
        )
    dev_datapoints = None
    if corpus.dev is not None and len(corpus.dev):
        dev_datapoints = corpus.dev.datapoints
        if hide_eval_sets_from_trainer:
            corpus._dev = None
    test_datapoints = None
    if corpus.test is not None and len(corpus.test):
        test_datapoints = corpus.test.datapoints
        if hide_eval_sets_from_trainer:
            corpus._test = None

    if use_lora:
        if freeze_embedding and _log is not None:
            _log.info("use_lora=True so freeze_embedding will be ignored")
        embedding = TransformerDocumentEmbeddingsWithLoRA(
            LoraConfig(target_modules=lora_target),
            save_only_lora_parameters,
            model=model_name,
        )
    else:
        embedding = TransformerDocumentEmbeddings(
            model_name, fine_tune=not freeze_embedding
        )
    if not embedding.tokenizer.pad_token:
        assert model_name not in ("roberta-base", "Twitter/twhin-bert-base")
        embedding.tokenizer.pad_token = embedding.tokenizer.eos_token
    model = corpus.make_classifier(embedding, label_dict)
    if method == "AEh":
        model.averages_probs = False
    elif method == "SmF1":
        kwargs = {} if corpus.multi_label else {"activation": "softmax"}
        model.loss_function = SoftMicroF1Loss(**kwargs)
    elif method == "SmF1-v2":
        model.loss_function = SoftMicroF1Loss(use_torch_minimum=False)
    elif method == "SMF1":
        kwargs = {} if corpus.multi_label else {"activation": "softmax"}
        model.loss_function = SoftMacroF1Loss(**kwargs)
    elif method == "JSD":
        model.loss_function = MultilabelJSDLoss() if corpus.multi_label else JSDLoss()
    elif method == "LA-max":
        model.loss_function.aggfunc = "max"
    elif method == "LA-mean":
        model.loss_function.aggfunc = "mean"
    trainer = ModelTrainer(model, corpus)
    plugins = []
    if dev_datapoints is not None and eval_on_epoch_finished:
        plugins.append(
            EvaluateOnDevPlugin(
                dev_datapoints, label_dict, artifacts_dir, test_batch_size, _log, _run
            )
        )
    if _run is not None:
        plugins.append(SacredLogMetricsPlugin(_run, log_every))
    if attach_oom_observer:
        torch.cuda.memory._record_memory_history()

        # Adapted from https://zdevito.github.io/2022/08/16/memory-snapshots.html
        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            if _log is not None:
                _log.info("Saving allocated state during OOM")
            torch.cuda.memory._dump_snapshot("oom_snapshot.pkl")

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    trainer.fine_tune(
        artifacts_dir,
        learning_rate=lr,
        mini_batch_size=batch_size,
        mini_batch_chunk_size=batch_chunk_size,
        eval_batch_size=test_batch_size,
        max_epochs=max_epochs,
        plugins=plugins,
        shuffle=not sort_train_desc,
    )
    if not save_model:
        (artifacts_dir / "final-model.pt").unlink()

    retval = (
        None
        if dev_datapoints is None or not eval_on_epoch_finished
        else plugins[0].retval
    )

    if dev_datapoints is not None and not eval_on_epoch_finished:
        retval = do_evaluation(
            model,
            label_dict,
            dev_datapoints,
            corpus.soft_label_type,
            artifacts_dir,
            test_batch_size,
            corpus.multi_label,
            is_test_set=False,
            _run=_run,
            _log=_log,
        )

    if test_datapoints is not None:
        do_evaluation(
            model,
            label_dict,
            test_datapoints,
            corpus.soft_label_type,
            artifacts_dir,
            test_batch_size,
            corpus.multi_label,
            _run=_run,
            _log=_log,
        )

    if retval is not None:
        retval = float(retval)
    return retval
