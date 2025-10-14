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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from flair.data import Dictionary
from sacred import Experiment
from sacred.observers import MongoObserver

from evaluation import entropy_correlation as ent_corr
from evaluation import hard_accuracy as hard_acc
from evaluation import hard_micro_f1
from evaluation import hard_precision_recall_fscore_support as hard_prfs
from evaluation import poJSD
from evaluation import soft_accuracy as soft_acc
from evaluation import soft_micro_f1
from evaluation import soft_precision_recall_fscore_support as soft_prfs

ex = Experiment("hlv-metrics")
if "SACRED_MONGO_URL" in os.environ:
    ex.observers.append(
        MongoObserver(
            url=os.environ["SACRED_MONGO_URL"],
            db_name=os.getenv("SACRED_DB_NAME", "sacred"),
        )
    )


@ex.config
def default():
    # directory where hlv.npy, label.dict, and inputs.jsonl are stored
    artifacts_dir = "artifacts"
    # test file (if true HLV isn't saved in hlv.npy)
    test_file = ""
    # whether the task is multilabel classification
    multilabel = False


def _compute_true_hlv(
    test_file: Path, inputs_file: Path, label_dict: Dictionary
) -> np.ndarray:
    input2counter: defaultdict[str | tuple[str, str], Counter[str]] = defaultdict(
        Counter
    )
    with test_file.open(encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            inp = obj["input"]
            try:
                text = inp["premise"]
            except KeyError:
                input2counter[inp["text"]][str(obj["output"]["class"])] += 1
            else:
                input2counter[text, inp["hypothesis"]][obj["output"]["verdict"]] += 1

    true_hlv = []
    with inputs_file.open(encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            try:
                text = obj["premise"]
            except KeyError:
                cntr = input2counter[obj["text"]]
            else:
                cntr = input2counter[text, obj["hypothesis"]]
            total = sum(cntr.values())
            true_hlv.append(
                [
                    cntr[label_dict.get_item_for_index(idx)] / total
                    for idx in range(len(label_dict))
                ]
            )
    return np.array(true_hlv)


@ex.capture
def _record_metric(name, value, fmt="%.3f", _run=None, _log=None):
    if _log is not None:
        _log.info(f"%s: {fmt}", name, value)
    if _run is not None:
        _run.log_scalar(name, value)


@ex.automain
def evaluate(artifacts_dir, test_file=None, multilabel=False, _run=None, _log=None):
    artifacts_dir = Path(artifacts_dir)
    label_dict = Dictionary.load_from_file(artifacts_dir / "label.dict")

    y_pred = np.load(artifacts_dir / "hlv.npy")
    if test_file:
        y_true = _compute_true_hlv(
            Path(test_file), artifacts_dir / "inputs.jsonl", label_dict
        )
    else:
        y_true, y_pred = y_pred

    if multilabel:
        overall_metrics = [
            hard_micro_f1,
            soft_micro_f1,
            lambda x, y: poJSD(x, y, multilabel=True),
        ]
        metric_names = "hard_micro_f1 soft_micro_f1 poJSD".split()
    else:
        overall_metrics = [hard_acc, soft_acc, poJSD]
        metric_names = "hard_acc soft_acc poJSD".split()
    overall_metrics.append(ent_corr)
    metric_names.append("ent_corr")
    for name, fn in zip(metric_names, overall_metrics):
        _record_metric(name, fn(y_true, y_pred), _run=_run)
    prfs = hard_prfs(y_true, y_pred)
    for i, x in enumerate("p r f1 supp".split()):
        if x != "supp":
            _record_metric(f"macro/hard_{x}", prfs[i].mean(), _run=_run)
        for j in range(len(prfs[i])):
            l = label_dict.get_item_for_index(j)
            fmt = "%d" if x == "supp" else "%.3f"
            _record_metric(f"hard_{x}/{l}", prfs[i][j], fmt, _run)
    prfs = soft_prfs(y_true, y_pred)
    for i, x in enumerate("p r f1 supp".split()):
        if x != "supp":
            _record_metric(f"macro/soft_{x}", prfs[i].mean(), _run=_run)
        for j in range(len(prfs[i])):
            l = label_dict.get_item_for_index(j)
            fmt = "%.2f" if x == "supp" else "%.3f"
            _record_metric(f"soft_{x}/{l}", prfs[i][j], fmt, _run)
