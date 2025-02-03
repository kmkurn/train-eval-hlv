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

import math
import warnings

import numpy as np
import pytest
from evaluation import (
    entropy_correlation,
    hard_accuracy,
    hard_micro_f1,
    hard_precision_recall_fscore_support,
    poJSD,
    soft_accuracy,
    soft_micro_f1,
    soft_precision_recall_fscore_support,
)
from scipy.stats import beta, dirichlet


@pytest.fixture(scope="module", params=[2, 3])
def dirichlet_rv(request):
    return dirichlet(np.full(request.param, fill_value=1.0), seed=0)


class FixedSizeRV:
    def __init__(self, beta_rv, size):
        self._beta_rv = beta_rv
        self._size = size

    def rvs(self, size):
        try:
            shape = list(size)
        except TypeError:
            shape = [size]
        shape.append(self._size)
        return self._beta_rv.rvs(shape)


@pytest.fixture(scope="module", params=[2, 3])
def beta_rv(request):
    rv = beta(a=1.0, b=1.0)
    rv.random_state = np.random.default_rng(seed=0)
    return FixedSizeRV(rv, request.param)


@pytest.fixture(scope="module")
def get_rv(dirichlet_rv, beta_rv):
    def _get_rv(multilabel=False):
        return beta_rv if multilabel else dirichlet_rv

    return _get_rv


def make_bindist(ps: np.ndarray) -> np.ndarray:
    return np.stack([ps, 1 - ps], axis=1)


def test_hard_acc_correct():
    ps = np.array([0.2, 0.6, 0.3])
    qs = np.array([0.4, 0.1, 0.1])
    res = hard_accuracy(make_bindist(ps), make_bindist(qs))

    assert res == pytest.approx(2 / 3)


def test_pojsd_correct():
    p, q = 0.2, 0.6
    res = poJSD(np.array([[p, 1 - p]]), np.array([[q, 1 - q]]))

    assert res == pytest.approx(0.8754887502)


def test_ent_corr_correct():
    p0, p1, p2 = 0.2, 0.6, 0.9
    q0, q1, q2 = 0.3, 0.5, 0.4
    ent_p0 = -(p0 * math.log2(p0) + (1 - p0) * math.log2(1 - p0))
    ent_p1 = -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1))
    ent_p2 = -(p2 * math.log2(p2) + (1 - p2) * math.log2(1 - p2))
    mean_ent_p = (ent_p0 + ent_p1 + ent_p2) / 3
    ent_q0 = -(q0 * math.log2(q0) + (1 - q0) * math.log2(1 - q0))
    ent_q1 = -(q1 * math.log2(q1) + (1 - q1) * math.log2(1 - q1))
    ent_q2 = -(q2 * math.log2(q2) + (1 - q2) * math.log2(1 - q2))
    mean_ent_q = (ent_q0 + ent_q1 + ent_q2) / 3
    ent_p0_z = ent_p0 - mean_ent_p
    ent_p1_z = ent_p1 - mean_ent_p
    ent_p2_z = ent_p2 - mean_ent_p
    ent_q0_z = ent_q0 - mean_ent_q
    ent_q1_z = ent_q1 - mean_ent_q
    ent_q2_z = ent_q2 - mean_ent_q
    num = ent_p0_z * ent_q0_z + ent_p1_z * ent_q1_z + ent_p2_z * ent_q2_z
    rssq_p = math.sqrt(ent_p0_z**2 + ent_p1_z**2 + ent_p2_z**2)
    rssq_q = math.sqrt(ent_q0_z**2 + ent_q1_z**2 + ent_q2_z**2)
    denom = rssq_p * rssq_q
    expected = num / denom

    res = entropy_correlation(
        np.array([[p0, 1 - p0], [p1, 1 - p1], [p2, 1 - p2]]),
        np.array([[q0, 1 - q0], [q1, 1 - q1], [q2, 1 - q2]]),
    )

    assert res == pytest.approx(expected)


def test_soft_acc_correct():
    p, q = 0.2, 0.6
    res = soft_accuracy(np.array([[p, 1 - p]]), np.array([[q, 1 - q]]))

    assert res == pytest.approx(0.6)


def test_hard_prfs_correct():
    ps = np.array([0.2, 0.3, 0.3, 0.9])
    qs = np.array([0.4, 0.1, 0.7, 0.8])

    Ps, Rs, Fs, Ns = hard_precision_recall_fscore_support(
        make_bindist(ps), make_bindist(qs)
    )

    assert Ps == pytest.approx([1 / 2, 1])
    assert Rs == pytest.approx([1, 2 / 3])
    assert Fs == pytest.approx([2 / 3, 4 / 5])
    assert Ns == pytest.approx([1, 3])


def test_hard_prfs_incomplete_mv_labels():
    p = np.array([[0.2, 0.8]])
    q = np.array([[0.1, 0.9]])

    res = hard_precision_recall_fscore_support(p, q)

    assert all([len(r) == 2 for r in res])


def test_soft_prfs_correct():
    ps = np.array([0.2, 0.3, 0.3, 0.9])
    qs = np.array([0.4, 0.1, 0.7, 0.8])

    Ps, Rs, Fs, Ns = soft_precision_recall_fscore_support(
        make_bindist(ps), make_bindist(qs)
    )

    assert Ps == pytest.approx([1.4 / 2.0, 1.7 / 2.0])
    assert Rs == pytest.approx([1.4 / 1.7, 1.7 / 2.3])
    assert Fs == pytest.approx([2 * 1.4 / 3.7, 2 * 1.7 / 4.3])
    assert Ns == pytest.approx([1.7, 2.3])


def test_hard_micro_f1_correct():
    ps = np.array([[0.2, 0.3], [0.1, 0.8], [0.9, 0.7]])
    qs = np.array([[0.8, 0.1], [0.3, 0.4], [0.7, 0.4]])
    res = hard_micro_f1(ps, qs)

    assert res == pytest.approx(2 / 5)


def test_soft_micro_f1_correct():
    ps = np.array([[0.2, 0.3], [0.1, 0.8], [0.9, 0.7]])
    qs = np.array([[0.8, 0.1], [0.3, 0.4], [0.7, 0.4]])
    res = soft_micro_f1(ps, qs)

    assert res == pytest.approx(
        2 * (0.2 + 0.1 + 0.1 + 0.4 + 0.7 + 0.4) / (1.0 + 0.4 + 0.4 + 1.2 + 1.6 + 1.1)
    )


def test_multilabel_pojsd_correct():
    p0, p1 = 0.2, 0.3
    q0, q1 = 0.9, 0.4
    res = poJSD(np.array([[p0, p1]]), np.array([[q0, q1]]), multilabel=True)
    m0 = np.array([(p0 + q0), (1 - p0 + 1 - q0)]) / 2
    m1 = np.array([(p1 + q1), (1 - p1 + 1 - q1)]) / 2
    kl_p0 = p0 * math.log2(p0 / m0[0]) + (1 - p0) * math.log2((1 - p0) / m0[1])
    kl_p1 = p1 * math.log2(p1 / m1[0]) + (1 - p1) * math.log2((1 - p1) / m1[1])
    kl_q0 = q0 * math.log2(q0 / m0[0]) + (1 - q0) * math.log2((1 - q0) / m0[1])
    kl_q1 = q1 * math.log2(q1 / m1[0]) + (1 - q1) * math.log2((1 - q1) / m1[1])
    jsd0 = (kl_p0 + kl_q0) / 2
    jsd1 = (kl_p1 + kl_q1) / 2
    expected = 1 - (jsd0 + jsd1) / 2

    assert res == pytest.approx(expected)


def test_multilabel_ent_corr_correct():
    p = np.array([[0.2, 0.9], [0.5, 0.7], [0.1, 0.1]])
    q = np.array([[0.1, 0.3], [0.8, 0.9], [0.7, 0.7]])
    ent_p = np.array(
        [
            [
                -(
                    p[0, 0] * math.log2(p[0, 0])
                    + (1 - p[0, 0]) * math.log2(1 - p[0, 0])
                ),
                -(
                    p[0, 1] * math.log2(p[0, 1])
                    + (1 - p[0, 1]) * math.log2(1 - p[0, 1])
                ),
            ],
            [
                -(
                    p[1, 0] * math.log2(p[1, 0])
                    + (1 - p[1, 0]) * math.log2(1 - p[1, 0])
                ),
                -(
                    p[1, 1] * math.log2(p[1, 1])
                    + (1 - p[1, 1]) * math.log2(1 - p[1, 1])
                ),
            ],
            [
                -(
                    p[2, 0] * math.log2(p[2, 0])
                    + (1 - p[2, 0]) * math.log2(1 - p[2, 0])
                ),
                -(
                    p[2, 1] * math.log2(p[2, 1])
                    + (1 - p[2, 1]) * math.log2(1 - p[2, 1])
                ),
            ],
        ]
    )
    ent_q = np.array(
        [
            [
                -(
                    q[0, 0] * math.log2(q[0, 0])
                    + (1 - q[0, 0]) * math.log2(1 - q[0, 0])
                ),
                -(
                    q[0, 1] * math.log2(q[0, 1])
                    + (1 - q[0, 1]) * math.log2(1 - q[0, 1])
                ),
            ],
            [
                -(
                    q[1, 0] * math.log2(q[1, 0])
                    + (1 - q[1, 0]) * math.log2(1 - q[1, 0])
                ),
                -(
                    q[1, 1] * math.log2(q[1, 1])
                    + (1 - q[1, 1]) * math.log2(1 - q[1, 1])
                ),
            ],
            [
                -(
                    q[2, 0] * math.log2(q[2, 0])
                    + (1 - q[2, 0]) * math.log2(1 - q[2, 0])
                ),
                -(
                    q[2, 1] * math.log2(q[2, 1])
                    + (1 - q[2, 1]) * math.log2(1 - q[2, 1])
                ),
            ],
        ]
    )
    mean_ent_p = (
        np.array(
            [
                ent_p[0, 0] + ent_p[1, 0] + ent_p[2, 0],
                ent_p[0, 1] + ent_p[1, 1] + ent_p[2, 1],
            ]
        )
        / 3
    )
    mean_ent_q = (
        np.array(
            [
                ent_q[0, 0] + ent_q[1, 0] + ent_q[2, 0],
                ent_q[0, 1] + ent_q[1, 1] + ent_q[2, 1],
            ]
        )
        / 3
    )
    ent_p_z = np.array(
        [
            [ent_p[0, 0] - mean_ent_p[0], ent_p[0, 1] - mean_ent_p[1]],
            [ent_p[1, 0] - mean_ent_p[0], ent_p[1, 1] - mean_ent_p[1]],
            [ent_p[2, 0] - mean_ent_p[0], ent_p[2, 1] - mean_ent_p[1]],
        ]
    )
    ent_q_z = np.array(
        [
            [ent_q[0, 0] - mean_ent_q[0], ent_q[0, 1] - mean_ent_q[1]],
            [ent_q[1, 0] - mean_ent_q[0], ent_q[1, 1] - mean_ent_q[1]],
            [ent_q[2, 0] - mean_ent_q[0], ent_q[2, 1] - mean_ent_q[1]],
        ]
    )
    num = [
        ent_p_z[0, 0] * ent_q_z[0, 0]
        + ent_p_z[1, 0] * ent_q_z[1, 0]
        + ent_p_z[2, 0] * ent_q_z[2, 0],
        ent_p_z[0, 1] * ent_q_z[0, 1]
        + ent_p_z[1, 1] * ent_q_z[1, 1]
        + ent_p_z[2, 1] * ent_q_z[2, 1],
    ]
    rssq_p = np.sqrt(
        np.array(
            [
                ent_p_z[0, 0] ** 2 + ent_p_z[1, 0] ** 2 + ent_p_z[2, 0] ** 2,
                ent_p_z[0, 1] ** 2 + ent_p_z[1, 1] ** 2 + ent_p_z[2, 1] ** 2,
            ]
        )
    )
    rssq_q = np.sqrt(
        np.array(
            [
                ent_q_z[0, 0] ** 2 + ent_q_z[1, 0] ** 2 + ent_q_z[2, 0] ** 2,
                ent_q_z[0, 1] ** 2 + ent_q_z[1, 1] ** 2 + ent_q_z[2, 1] ** 2,
            ]
        )
    )
    denom = [rssq_p[0] * rssq_q[0], rssq_p[1] * rssq_q[1]]
    expected = (num[0] / denom[0] + num[1] / denom[1]) / 2

    res = entropy_correlation(p, q, multilabel=True)

    assert res == pytest.approx(expected)


def test_multilabel_hard_prfs_correct():
    p = np.array([[0.1, 0.2], [0.2, 0.7], [0.8, 0.9], [0.8, 0.2]])
    q = np.array([[0.9, 0.3], [0.9, 0.2], [0.1, 0.6], [0.7, 0.8]])

    Ps, Rs, Fs, Ns = hard_precision_recall_fscore_support(p, q, multilabel=True)

    assert Ps == pytest.approx([1 / 3, 1 / 2])
    assert Rs == pytest.approx([1 / 2, 1 / 2])
    assert Fs == pytest.approx([2 / 5, 1 / 2])
    assert Ns == pytest.approx([2, 2])


num_repeats = 100
prf_fns = [lambda p, q: hard_precision_recall_fscore_support(p, q)[i] for i in range(3)]
mul_prf_fns = [
    lambda p, q: hard_precision_recall_fscore_support(p, q, multilabel=True)[i]
    for i in range(3)
]
sprf_fns = [
    lambda p, q: soft_precision_recall_fscore_support(p, q)[i] for i in range(3)
]


@pytest.mark.slow
@pytest.mark.repeat(num_repeats)
@pytest.mark.parametrize("n_samples", [2, 5000])
@pytest.mark.parametrize(
    "eval_fn,lower,upper,multilabel",
    list(
        zip(
            [
                hard_accuracy,
                poJSD,
                entropy_correlation,
                soft_accuracy,
                *prf_fns,
                *sprf_fns,
                hard_micro_f1,
                soft_micro_f1,
                lambda p, q: poJSD(p, q, multilabel=True),
                lambda p, q: entropy_correlation(p, q, multilabel=True),
                *mul_prf_fns,
            ],
            [0, 0, -1] + [0] * 10 + [-1] + [0] * 3,
            [1, 1, 1, poJSD] + [1] * 13,
            [False] * 10 + [True] * 7,
        )
    ),
)
def test_bounded(get_rv, n_samples, eval_fn, lower, upper, multilabel):
    p, q = get_rv(multilabel).rvs(size=[2, n_samples])
    if callable(upper):
        upper = upper(p, q)
    res = eval_fn(p, q)
    if isinstance(res, np.ndarray):
        assert ((lower <= res) & (res <= upper)).all()
    else:
        assert lower <= res <= upper


@pytest.mark.slow
@pytest.mark.repeat(num_repeats)
@pytest.mark.parametrize("n_samples", [2, 5000])
@pytest.mark.parametrize(
    "eval_fn,multilabel",
    list(
        zip(
            [
                hard_accuracy,
                poJSD,
                entropy_correlation,
                soft_accuracy,
                *prf_fns,
                *sprf_fns,
                hard_micro_f1,
                soft_micro_f1,
                lambda p, q: poJSD(p, q, multilabel=True),
                lambda p, q: entropy_correlation(p, q, multilabel=True),
                *mul_prf_fns,
            ],
            [False] * 10 + [True] * 7,
        )
    ),
)
def test_oracle(get_rv, n_samples, eval_fn, multilabel):
    p = get_rv(multilabel).rvs(n_samples)
    assert eval_fn(p, p) == pytest.approx(1.0)


@pytest.mark.parametrize("multilabel", [False, True])
def test_ent_corr_zero_prob(multilabel):
    p1, q1 = np.array([1.0, 0.3, 0.2]), np.array([0.3, 0.4, 0.1])
    p2, q2 = np.array([0.4, 0.9, 0.8]), np.array([0.7, 1.0, 0.9])

    def ent_corr(p, q):
        if multilabel:
            p_, q_ = p[..., None], q[..., None]
        else:
            p_, q_ = make_bindist(p), make_bindist(q)
        return entropy_correlation(p_, q_, multilabel)

    ent_corr(p1, q1)
    ent_corr(p2, q2)


@pytest.mark.parametrize("multilabel", [False, True])
def test_pojsd_zero_prob(multilabel):
    p1, q1 = np.array([1.0, 0.3]), np.array([0.3, 0.4])
    p2, q2 = np.array([0.4, 0.9]), np.array([0.7, 1.0])
    p3, q3 = np.array([1.0, 0.9]), np.array([1.0, 0.3])

    def pojsd(p, q):
        if multilabel:
            p_, q_ = p[None, ...], q[None, ...]
        else:
            p_, q_ = make_bindist(p), make_bindist(q)
        return poJSD(p_, q_, multilabel)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res1 = pojsd(p1, q1)
        res2 = pojsd(p2, q2)
        res3 = pojsd(p3, q3)

    assert not np.isnan(res1)
    assert not np.isnan(res2)
    assert not np.isnan(res3)


@pytest.mark.slow
@pytest.mark.repeat(num_repeats)
@pytest.mark.parametrize("n_samples", [2, 5000])
def test_soft_micro_f1_equal_soft_acc_when_unity(get_rv, n_samples):
    p, q = get_rv(multilabel=False).rvs(size=[2, n_samples])
    assert soft_micro_f1(p, q) == pytest.approx(soft_accuracy(p, q))


@pytest.mark.slow
@pytest.mark.repeat(num_repeats)
@pytest.mark.parametrize("n_samples", [2, 5000])
def test_multilabel_soft_prf_always_equal_multiclass(get_rv, n_samples):
    p, q = get_rv(multilabel=True).rvs(size=[2, n_samples])
    mul_prfs = soft_precision_recall_fscore_support(p, q, multilabel=True)
    muc_prfs = soft_precision_recall_fscore_support(p, q, multilabel=False)
    assert list(mul_prfs[:3]) == [pytest.approx(x) for x in muc_prfs[:3]]
