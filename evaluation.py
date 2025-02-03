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

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

Arr = np.ndarray


def hard_accuracy(y_true: Arr, y_pred: Arr) -> float:
    return accuracy_score(_mv(y_true), _mv(y_pred))


def hard_micro_f1(y_true: Arr, y_pred: Arr) -> float:
    denom = (y_true > 0.5).sum() + (y_pred > 0.5).sum()
    return (2 * ((y_true > 0.5) & (y_pred > 0.5)).sum() / denom) if denom else 1.0


def poJSD(y_true: Arr, y_pred: Arr, multilabel: bool = False) -> float:
    if multilabel:
        y_true = np.stack([y_true, 1 - y_true], axis=-1)
        y_pred = np.stack([y_pred, 1 - y_pred], axis=-1)
    y_mean = 0.5 * (y_true + y_pred)
    y_true_div_mean = np.divide(
        y_true, y_mean, out=np.zeros_like(y_true), where=~np.isclose(y_mean, 0)
    )
    y_pred_div_mean = np.divide(
        y_pred, y_mean, out=np.zeros_like(y_pred), where=~np.isclose(y_mean, 0)
    )
    kl_true_mean = (y_true * _log_or_zero(y_true_div_mean, np.log2)).sum(axis=-1)
    kl_pred_mean = (y_pred * _log_or_zero(y_pred_div_mean, np.log2)).sum(axis=-1)
    return 1 - (0.5 * (kl_true_mean + kl_pred_mean)).mean()


def entropy_correlation(y_true: Arr, y_pred: Arr, multilabel: bool = False) -> float:
    if multilabel:
        y_true = np.stack([y_true, 1 - y_true], axis=-1)
        y_pred = np.stack([y_pred, 1 - y_pred], axis=-1)
    ent_y_true = -(y_true * _log_or_zero(y_true)).sum(axis=-1)
    norm_ent_y_true = ent_y_true / np.log(y_true.shape[-1])
    ent_y_pred = -(y_pred * _log_or_zero(y_pred)).sum(axis=-1)
    norm_ent_y_pred = ent_y_pred / np.log(y_pred.shape[-1])
    if multilabel:
        return np.mean(
            [
                pearsonr(norm_ent_y_true[:, k], norm_ent_y_pred[:, k]).statistic
                for k in range(y_true.shape[1])
            ]
        )
    return pearsonr(norm_ent_y_true, norm_ent_y_pred).statistic


def soft_accuracy(y_true: Arr, y_pred: Arr) -> float:
    return np.where(y_true < y_pred, y_true, y_pred).sum(axis=-1).mean()


def soft_micro_f1(y_true: Arr, y_pred: Arr) -> float:
    return 2 * np.where(y_true < y_pred, y_true, y_pred).sum() / (y_true + y_pred).sum()


def hard_precision_recall_fscore_support(
    y_true: Arr,
    y_pred: Arr,
    multilabel: bool = False,
) -> tuple[Arr, Arr, Arr, Arr]:
    K = y_true.shape[-1]
    if multilabel:
        y_true = y_true > 0.5
        y_pred = y_pred > 0.5
    else:
        y_true = _mv(y_true)
        y_pred = _mv(y_pred)
    return precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(K), zero_division=1
    )


def soft_precision_recall_fscore_support(
    y_true: Arr, y_pred: Arr, multilabel: bool = False,
) -> tuple[Arr, Arr, Arr, Arr]:
    # multilabel is ignored as it has equal definition to multiclass
    min_sum = np.where(y_true < y_pred, y_true, y_pred).sum(axis=0)
    pred_sum = y_pred.sum(axis=0)
    true_sum = y_true.sum(axis=0)
    Ps = min_sum / pred_sum
    Rs = min_sum / true_sum
    Fs = 2 * min_sum / (pred_sum + true_sum)
    return Ps, Rs, Fs, true_sum


def _mv(p: Arr, /) -> Arr:
    return p.argmax(axis=-1)


def _log_or_zero(x: Arr, /, log_fn=np.log) -> Arr:
    atol = 1e-8
    return log_fn(x, out=np.zeros_like(x), where=x >= atol)
