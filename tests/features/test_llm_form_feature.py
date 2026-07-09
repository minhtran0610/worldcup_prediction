from __future__ import annotations

from dataclasses import dataclass

import pytest

from features.llm_form_feature import (
    K_TRAJECTORY,
    apply_trajectory_adjustment,
    compute_trajectory_factor,
)


@dataclass
class _FakeAnalysis:
    form_score: float
    confidence: float


def test_compute_trajectory_factor_below_min_confidence_returns_one():
    analysis = _FakeAnalysis(form_score=0.9, confidence=0.1)  # below default min_confidence
    assert compute_trajectory_factor(analysis) == pytest.approx(1.0)


def test_compute_trajectory_factor_scales_by_form_score():
    analysis = _FakeAnalysis(form_score=0.5, confidence=0.8)
    expected = 1.0 + K_TRAJECTORY * 0.5
    assert compute_trajectory_factor(analysis) == pytest.approx(expected, rel=1e-6)


def test_compute_trajectory_factor_clamped_to_range():
    analysis = _FakeAnalysis(form_score=1.0, confidence=1.0)
    factor = compute_trajectory_factor(analysis)
    assert 0.85 <= factor <= 1.15


def test_apply_trajectory_adjustment_scales_lambdas_and_passes_rho_through():
    lh_adj, la_adj, rho_adj = apply_trajectory_adjustment(2.0, 1.0, 0.05, 1.10, 0.90)
    assert lh_adj == pytest.approx(2.2)
    assert la_adj == pytest.approx(0.9)
    assert rho_adj == 0.05
