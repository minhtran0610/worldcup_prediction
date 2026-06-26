"""Apply LLM narrative form scores as post-hoc λ adjustments.

The neural model predicts. The LLM reads news articles. This module bridges them:
it translates a FormAnalysis (form_score, confidence) into a λ multiplier that
trims the neural model's Poisson rates up or down.

Formula
-------
    adj_factor = 1 + K_SENTIMENT * form_score * confidence
    lambda_adj  = lambda_base * adj_factor

K_SENTIMENT default: 0.30
    At full confidence and extreme score (±1.0), this is a ±30% λ shift —
    deliberately on par with the injury adjustment's realistic peak. Narrative
    form (tournament momentum, performance-vs-result, morale, manager situation)
    captures team-level signal that confirmed absences alone miss, so it is given
    equal authority. The safeguard against narrative noise is *confidence*, not a
    small coefficient: the shift scales with the LLM's extraction confidence, so a
    thin or hedgy read moves λ almost nothing while a richly-sourced, decisive read
    can move it as much as a major injury.

Confidence gating
-----------------
    Analyses with confidence < MIN_CONFIDENCE are treated as zero signal.
    Default MIN_CONFIDENCE = 0.25. This guards against:
      - Articles that mention the team only in passing
      - LLM hedging when the text is ambiguous
      - Low-information article bodies (video/gallery pages)

Application order
-----------------
    Apply injury adjustment first (more mechanistic, better trusted), then
    sentiment. In log space they are additive:
        log λ_final = log λ_base + log(1 - K_inj * loss) + log(1 + K_sent * score * conf)
"""

from __future__ import annotations

K_SENTIMENT: float = 0.30
MIN_CONFIDENCE: float = 0.25


def compute_sentiment_factor(
    form_score: float,
    confidence: float,
    k: float = K_SENTIMENT,
    min_confidence: float = MIN_CONFIDENCE,
) -> float:
    """Return the λ multiplier for one team given its narrative form score.

    Parameters
    ----------
    form_score:
        Narrative form score in [-1, 1] from FormAnalysis.
    confidence:
        Extraction confidence in [0, 1] from FormAnalysis.
    k:
        Dampening coefficient. Default 0.30 — max ±30% λ shift at full confidence,
        on par with the injury adjustment.
    min_confidence:
        Analyses below this threshold are treated as no signal (factor = 1.0).

    Returns
    -------
    float
        Multiplier to apply to λ.  Range ≈ [0.70, 1.30] at default K.
        Returns 1.0 (no change) when confidence < min_confidence.
    """
    if confidence < min_confidence:
        return 1.0
    factor = 1.0 + k * form_score * confidence
    # Hard floor / ceiling to prevent pathological values from bad LLM output
    return max(0.70, min(1.30, factor))


def apply_sentiment_adjustment(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    form_score_home: float,
    form_score_away: float,
    confidence_home: float,
    confidence_away: float,
    k: float = K_SENTIMENT,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[float, float, float]:
    """Return (lambda_home_adj, lambda_away_adj, rho) after narrative form scaling.

    rho is passed through unchanged — it is a match-context correlation term,
    not a goal-rate term, so it does not scale with form signals.

    Apply AFTER injury adjustment so that the more mechanistic signal takes
    precedence:
        lh, la, rho = apply_injury_adjustment(...)
        lh, la, rho = apply_sentiment_adjustment(lh, la, rho, ...)

    Parameters
    ----------
    lambda_home, lambda_away:
        Poisson rates, already injury-adjusted if applicable.
    rho:
        Dixon-Coles correlation, passed through.
    form_score_home, form_score_away:
        Narrative form scores in [-1, 1] per team from FormAnalysis.
    confidence_home, confidence_away:
        Extraction confidence in [0, 1] per team.
    k, min_confidence:
        Forwarded to compute_sentiment_factor().
    """
    fh = compute_sentiment_factor(form_score_home, confidence_home, k, min_confidence)
    fa = compute_sentiment_factor(form_score_away, confidence_away, k, min_confidence)
    lh_adj = max(lambda_home * fh, 0.01)
    la_adj = max(lambda_away * fa, 0.01)
    return lh_adj, la_adj, rho


def build_sentiment_report_line(
    team: str,
    form_score: float,
    confidence: float,
    key_absences: list[str],
    factor: float,
    min_confidence: float = MIN_CONFIDENCE,
) -> str:
    """Format a one-line sentiment summary for CLI display.

    Examples
    --------
    England   form +0.20 conf 0.55  λ×1.009  "not afraid to shout" "always demanding"
    France    no signal (conf 0.08)  λ unchanged
    Cape Verde  no articles found
    """
    name = f"{team:<22}"
    if confidence < min_confidence:
        return f"  {name}  no signal (conf {confidence:.2f})  λ unchanged"

    direction = f"{form_score:+.2f}"
    factor_str = f"λ×{factor:.3f}"
    absences_str = f"  absent: {', '.join(key_absences)}" if key_absences else ""
    return f"  {name}  form {direction}  conf {confidence:.2f}  {factor_str}{absences_str}"
