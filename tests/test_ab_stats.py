import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
import ab_stats as ab  # noqa: E402


def test_sample_size_matches_reference_value():
    # Reference value: Evan Miller's calculator (p=10%, relative MDE 10%, alpha=0.05, power 80%)
    assert ab.sample_size_proportions(0.10, 0.10) == pytest.approx(14_750, rel=0.01)


def test_srm_detects_broken_split():
    assert ab.srm_check(5_000, 5_050) > 0.05
    assert ab.srm_check(5_000, 5_500) < 0.001


def test_ztest_no_difference():
    r = ab.two_proportion_ztest(500, 10_000, 500, 10_000)
    assert r.p_value == pytest.approx(1.0)
    assert r.ci_low < 0 < r.ci_high


def test_ztest_detects_large_difference():
    r = ab.two_proportion_ztest(500, 10_000, 650, 10_000)
    assert r.significant
    assert r.rel_lift == pytest.approx(0.30)
    assert r.ci_low > 0


def test_welch_and_bootstrap_agree():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 2_000)
    b = rng.normal(10.5, 2, 2_000)
    welch = ab.welch_ttest(a, b)
    lo, hi, _ = ab.bootstrap_mean_diff(a, b, n_boot=2_000)
    assert welch.ci_low == pytest.approx(lo, abs=0.05)
    assert welch.ci_high == pytest.approx(hi, abs=0.05)


def test_bayesian_probability_is_symmetric():
    res = ab.bayesian_proportions(500, 10_000, 500, 10_000)
    assert res["prob_b_better"] == pytest.approx(0.5, abs=0.01)


def test_holm_correction():
    adjusted = ab.holm_correction([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
