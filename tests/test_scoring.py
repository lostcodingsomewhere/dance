"""Unit tests for the shared scoring core (:mod:`dance.recommender.scoring`)."""

from __future__ import annotations

import numpy as np
import pytest

from dance.recommender import scoring as sc

# --- embed_score ----------------------------------------------------------


def test_embed_score_identical_is_one():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert sc.embed_score(v, v) == pytest.approx(1.0)


def test_embed_score_opposite_is_zero():
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert sc.embed_score(v, -v) == pytest.approx(0.0)


def test_embed_score_orthogonal_is_half():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert sc.embed_score(a, b) == pytest.approx(0.5)


def test_embed_score_none_inputs():
    assert sc.embed_score(None, np.array([1.0])) is None
    assert sc.embed_score(np.array([1.0]), None) is None


# --- key_score ------------------------------------------------------------


def test_key_score_exact():
    assert sc.key_score("8A", ["8A"]) == pytest.approx(1.0)


def test_key_score_compatible():
    # 8A is compatible with 8B (relative) and 7A/9A (adjacent).
    assert sc.key_score("8B", ["8A"]) == pytest.approx(0.6)


def test_key_score_incompatible():
    assert sc.key_score("2A", ["8A"]) == pytest.approx(0.0)


def test_key_score_none_when_no_data():
    assert sc.key_score(None, ["8A"]) is None
    assert sc.key_score("8A", []) is None


# --- bpm_score ------------------------------------------------------------


def test_bpm_score_exact():
    assert sc.bpm_score(128.0, 128.0) == pytest.approx(1.0)


def test_bpm_score_linear_falloff():
    # half the default tolerance away → 0.5
    assert sc.bpm_score(128.0 + sc.DEFAULT_BPM_TOL / 2, 128.0) == pytest.approx(0.5)


def test_bpm_score_beyond_tolerance_is_zero():
    assert sc.bpm_score(128.0 + sc.DEFAULT_BPM_TOL + 5, 128.0) == pytest.approx(0.0)


def test_bpm_score_halftime_match():
    # 70 against 140 should match via the doubled fold (×0.9 penalty).
    assert sc.bpm_score(70.0, 140.0) == pytest.approx(0.9, abs=1e-3)


def test_bpm_score_halftime_disabled():
    assert sc.bpm_score(70.0, 140.0, halftime=False) == pytest.approx(0.0)


def test_bpm_score_none():
    assert sc.bpm_score(None, 128.0) is None


# --- ratio / delta / derived ----------------------------------------------


def test_ratio_score():
    assert sc.ratio_score(2.0, 2.0) == pytest.approx(1.0)
    assert sc.ratio_score(1.0, 2.0) == pytest.approx(0.5)
    assert sc.ratio_score(0.0, 0.0) == pytest.approx(1.0)
    assert sc.ratio_score(None, 1.0) is None


def test_delta_score():
    assert sc.delta_score(0.5, 0.5) == pytest.approx(1.0)
    assert sc.delta_score(0.2, 0.7, scale=1.0) == pytest.approx(0.5)
    assert sc.delta_score(0.0, 1.0) == pytest.approx(0.0)


def test_timbre_score_averages_present_components():
    # brightness ratio 0.5, warmth ratio 1.0 → mean 0.75
    assert sc.timbre_score((1.0, 2.0), (2.0, 2.0)) == pytest.approx(0.75)
    # only one component comparable
    assert sc.timbre_score((1.0, None), (2.0, None)) == pytest.approx(0.5)
    assert sc.timbre_score((None, None), (None, None)) is None


# --- confidence_gate ------------------------------------------------------


def test_confidence_gate_none_confidence_passthrough():
    assert sc.confidence_gate(0.8, None) == pytest.approx(0.8)


def test_confidence_gate_discounts():
    assert sc.confidence_gate(0.8, 0.5) == pytest.approx(0.4)


def test_confidence_gate_none_raw():
    assert sc.confidence_gate(None, 0.5) is None


# --- project_energy -------------------------------------------------------


def test_project_energy_slope():
    assert sc.project_energy([3, 4, 5, 6]) == pytest.approx(7.0)


def test_project_energy_clamps():
    assert sc.project_energy([8, 9, 10, 10]) <= 10.0


def test_project_energy_edge_cases():
    assert sc.project_energy([]) is None
    assert sc.project_energy([7]) == pytest.approx(7.0)


# --- combine + profiles ---------------------------------------------------


def test_combine_renormalizes_over_present_features():
    prof = sc.Profile("t", {"embedding": 0.5, "key": 0.3, "bpm": 0.2})
    # Only embed + bpm present → renormalize over 0.7.
    score, breakdown = sc.combine({"embedding": 1.0, "key": None, "bpm": 0.0}, prof)
    assert score == pytest.approx(0.5 / 0.7)
    assert "key" not in breakdown
    assert breakdown["embedding"] == 1.0


def test_combine_all_missing_is_zero():
    prof = sc.Profile("t", {"embedding": 1.0})
    score, breakdown = sc.combine({"embedding": None}, prof)
    assert score == 0.0
    assert breakdown == {}


def test_combine_perfect_candidate_scores_one():
    prof = sc.PROFILES["column:mix"]
    feats = {"embedding": 1.0, "key": 1.0, "bpm": 1.0, "timbre": 1.0}
    score, _ = sc.combine(feats, prof)
    assert score == pytest.approx(1.0)


def test_drums_profile_excludes_key():
    prof = sc.profile_for_column("drums")
    assert "key" not in prof.weights
    assert "kick_density" in prof.weights


def test_profile_for_column_defaults_to_mix():
    assert sc.profile_for_column("nonsense").name == "column:mix"
