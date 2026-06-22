"""Compute injury-adjusted Poisson rate parameters for WC 2026 predictions.

The neural model outputs λ_home and λ_away from historical form/Elo/squad context.
Injuries are an exogenous shock applied on top: the model has no historical injury
labels, so learning injury weights is impossible. Instead we scale the Poisson rates:

    λ_home_adj = λ_home * (1 - K * inj_loss_home)
    λ_away_adj = λ_away * (1 - K * inj_loss_away)

where K ∈ [0, 1] is a dampening coefficient (default 0.5 — the market typically
prices a top-forward absence as roughly half their xG contribution, not the full
amount, because coaches compensate tactically).

inj_loss is the caps-weighted fraction of the squad that is absent:

    inj_loss = Σ caps_i (for absent players) / Σ caps_j (for full squad)

Caps proxy works well for international football: more-capped players are
the stronger players almost by definition in this context.

Name matching uses NFC Unicode normalisation + difflib fuzzy matching (cutoff 0.80)
to bridge API-Football names vs squad registry names.
"""

from __future__ import annotations

import difflib
import unicodedata

import numpy as np
import pandas as pd

DEFAULT_K: float = 0.5


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip().lower())


def _best_match(name: str, candidates: list[str], cutoff: float = 0.80) -> str | None:
    """Return the best fuzzy match for `name` in `candidates`, or None."""
    norm_name = _nfc(name)
    norm_candidates = [_nfc(c) for c in candidates]

    # Exact match first
    if norm_name in norm_candidates:
        return candidates[norm_candidates.index(norm_name)]

    # Fuzzy match on normalised strings
    matches = difflib.get_close_matches(norm_name, norm_candidates, n=1, cutoff=cutoff)
    if matches:
        return candidates[norm_candidates.index(matches[0])]

    # Last resort: substring — "Mbappé" matches "Kylian Mbappé"
    for i, cand in enumerate(norm_candidates):
        if norm_name in cand or cand in norm_name:
            return candidates[i]

    return None


def compute_injury_strength_loss(
    team: str,
    absent_players: list[str],
    squad_df: pd.DataFrame,
    fuzzy_cutoff: float = 0.80,
) -> float:
    """Return the caps-weighted fraction of squad strength that is absent.

    Parameters
    ----------
    team:
        Canonical team name (Kaggle training-data style).
    absent_players:
        Player names from the injury feed (API-Football or Transfermarkt).
    squad_df:
        Raw squad DataFrame from SquadRegistry._cache['wc2026'].
        Expected columns: team, player_name, caps.
    fuzzy_cutoff:
        Minimum similarity score for fuzzy name matching (0–1).

    Returns
    -------
    float in [0, 1]
        0.0 → no known absences (or squad data unavailable)
        0.3 → ~30% of caps-weighted squad strength absent
    """
    if not absent_players:
        return 0.0

    team_squad = squad_df[squad_df["team"] == team]
    if team_squad.empty:
        return 0.0

    caps = pd.to_numeric(team_squad["caps"], errors="coerce").fillna(0.0)
    total_caps = float(caps.sum())

    # Fall back to headcount weighting if all caps are zero
    if total_caps == 0.0:
        n = len(team_squad)
        matched = sum(
            1
            for p in absent_players
            if _best_match(p, team_squad["player_name"].tolist(), fuzzy_cutoff) is not None
        )
        return float(np.clip(matched / n, 0.0, 1.0)) if n > 0 else 0.0

    squad_names = team_squad["player_name"].tolist()
    absent_caps = 0.0
    matched_names: list[str] = []

    for absent in absent_players:
        match = _best_match(absent, squad_names, fuzzy_cutoff)
        if match is not None and match not in matched_names:
            matched_names.append(match)
            idx = team_squad[team_squad["player_name"] == match].index
            if len(idx) > 0:
                absent_caps += float(
                    pd.to_numeric(team_squad.loc[idx[0], "caps"], errors="coerce") or 0.0
                )

    return float(np.clip(absent_caps / total_caps, 0.0, 1.0))


def apply_injury_adjustment(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    inj_loss_home: float,
    inj_loss_away: float,
    k: float = DEFAULT_K,
) -> tuple[float, float, float]:
    """Return (lambda_home_adj, lambda_away_adj, rho) after applying injury scaling.

    rho is passed through unchanged — the Dixon-Coles correlation term is a
    property of the match context, not the goal rate, so it does not scale
    with squad availability.
    """
    lh_adj = lambda_home * (1.0 - k * inj_loss_home)
    la_adj = lambda_away * (1.0 - k * inj_loss_away)
    return max(lh_adj, 0.01), max(la_adj, 0.01), rho


def build_injury_report_line(
    team: str,
    absent_players: list[str],
    loss: float,
) -> str:
    """Format a one-line injury summary for CLI display."""
    if not absent_players:
        return f"  {team}: no known absences"
    names = ", ".join(absent_players[:5])
    suffix = f" (+{len(absent_players) - 5} more)" if len(absent_players) > 5 else ""
    return f"  {team}: {names}{suffix}  [strength loss {loss * 100:.1f}%]"
