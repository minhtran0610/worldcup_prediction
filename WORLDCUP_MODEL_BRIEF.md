# World Cup 2026 — Betting-Markets Prediction Model

**Project handoff brief for a Claude Code session.**
Rename to `CLAUDE.md` if you want Claude Code to load it automatically as project context.

---

## 0. How to use this document (read first)

You (Claude Code) are joining as a **discussion partner and implementation collaborator**, not a one-shot code generator. Before scaffolding anything:

1. Read this whole brief.
2. Work through **Section 9 (Open questions)** with me — these are unresolved design decisions. Ask me about them; don't silently pick defaults.
3. Only after we've agreed on the open items, propose a concrete first-commit plan (file tree + what each module does) and let me approve it before writing code.
4. Then build in the order given in Section 8, baselines first.

Push back on me. If a design choice here is weak, say so. The point of this project is to learn something true, not to confirm a pretty plan.

---

## 1. Context & goals

- **What it is:** a scientific hobby project to model football match outcomes and derive probabilities for betting markets (1X2, over/under, BTTS, correct score, Asian handicap), targeting the **2026 FIFA World Cup** (48 teams; hosts USA / Canada / Mexico; tournament starts mid-June 2026 — i.e. *very soon*, so live odds will be available shortly).
- **Why:** to learn deep-learning modelling end-to-end and explore whether a model can be well-calibrated and occasionally find value against the market.
- **What it is NOT:** an income scheme. Treat profit as a *diagnostic of calibration/edge*, not a goal. The realistic win is a well-calibrated model that sometimes finds value vs. the closing line.
- **Constraints:** pure Python, terminal/CLI interface only (no web UI). Single-developer, weekend-scale increments.
- **Hardware:** RTX 4070 Ti Super (16 GB VRAM), used locally. Note: the core model is tiny and GPU is overkill for one net — the GPU's real value is cheap hyperparameter sweeps, deep ensembles, and an optional richer per-player encoder.

---

## 2. The core design idea (do not skip)

**Do not build a separate model per market.** Build **one** model that predicts the **joint distribution over scorelines** — `P(home scores i, away scores j)` over a grid (e.g. 0–7 × 0–7). Every market is then a deterministic sum over that grid:

- Home win = Σ cells where `i > j`; Draw = `i = j`; Away = `i < j`
- Over 2.5 = Σ where `i + j ≥ 3`
- BTTS = Σ where `i ≥ 1 and j ≥ 1`
- Correct score / Asian handicap = appropriate sums

This gives **internally coherent** probabilities across all markets from a single forward pass.

The neural net's job: map match context → parameters of a goal distribution: `λ_home`, `λ_away`, and a low-score dependence term `ρ`. Use a **bivariate Poisson**, or **independent Poisson with a Dixon–Coles low-score (τ) correction** (a pure independent Poisson systematically misprices 0-0 / 1-0 / 0-1 / 1-1).

---

## 3. Honest caveats baked into the design

- **Data sparsity is the #1 risk.** Senior international football has only ~tens of thousands of matches ever; each national team plays ~10–15 competitive matches/year vs. wildly varying opposition. Deep learning will overfit unless heavily regularized and unless we borrow signal from abundant **club-level player data** (national teams = collections of club players).
- **Beating the closing line is hard.** Bookmaker closing odds (margin removed) are near-optimal. The honest yardstick is value vs. *closing* odds, not opening odds (beating opening odds usually just means you were slower than the market).
- **Gradient boosting may beat the neural net.** XGBoost/LightGBM often wins on tabular football data. If our net can't beat it, that's a *finding*, not a failure. The DL approach earns its place via learned team embeddings, a sequence form-encoder, and cheap ensembling on the GPU — not a guaranteed accuracy win.

---

## 4. Data sources (verified current as of June 2026)

**Match results — the training spine**
- Kaggle "International football results from 1872 to present" (Mart Jürisoo): ~45k+ internationals with score, tournament, neutral-venue flag, host.
- FIFA rankings over time (Kaggle / scrapeable): weak prior + feature.

**Player & club form — to fight sparsity**
- `soccerdata` (PyPI, v1.9, actively maintained): uniform pandas DataFrames from FBref, Understat, Club Elo, SoFIFA, Football-Data.co.uk. Explicitly covers men's & women's World Cups. Docs: https://soccerdata.readthedocs.io/
- Transfermarkt market values: strong squad-strength proxy.
- Map World Cup squads → their club-season xG/xA/minutes.

**Odds data — the scientific yardstick**
- `Football-Data.co.uk` (free, via soccerdata): historical *closing* odds for many leagues. Use to develop/validate the value-betting method on abundant club data first.
- `the-odds-api.com`: permanent free tier (~500 credits/month; historical calls cost 10× — budget carefully). For live World Cup match odds.
- `API-Football`: free soccer tier, broad coverage — alternative live-odds source.

**Explicitly dropped:** the live-tweet sentiment idea (X/API is paid/expensive now and adds little over market odds).

---

## 5. Feature engineering

All features computed **as-of the match date** — strictly no future leakage.

- **Rating priors:** a self-computed Elo (with inactivity decay) + FIFA rank. Elo alone is a strong baseline.
- **Recent form:** last-N matches' goals for/against, xG for/against where available, results — passed as a **sequence**, not just averages.
- **Squad strength:** aggregated club xG/90, minutes-weighted market value, share of squad in top-5 leagues.
- **Context:** home / away / **neutral** (most WC matches are neutral — important), host-nation flag, rest days, travel/altitude (optional), tournament stage (group vs. knockout shifts style & draw rate).
- **Head-to-head:** recent only, weak signal, include cautiously.

---

## 6. Model architecture

Three branches → fusion → distribution head:

```
Team-embedding branch:  home_id, away_id -> nn.Embedding (~16-32 dim each).
                        Neural generalization of Elo / Dixon-Coles strengths.

Form branch:            last-N match sequence per team -> small GRU or
                        1-layer Transformer encoder -> form context vector.

Context branch:         neutral/home/host/rest/stage/squad-value -> MLP.

Fusion -> MLP -> head:  concat -> MLP -> outputs:
                          lambda_home = softplus(...)
                          lambda_away = softplus(...)
                          rho         = tanh(...)   # low-score dependence
```

From `(λ_home, λ_away, ρ)`, build the full score-grid probability matrix analytically.

**Baselines to build FIRST and beat:**
1. Bookmaker implied probabilities, margin removed (the bar).
2. Plain Elo.
3. Dixon–Coles (no neural net).
4. XGBoost / LightGBM on the same features.

**Stretch (GPU-justifying) version:** per-player Transformer encoder — encode each of the 22 starters' club form, pool into team vectors.

---

## 7. Training methodology

- **Loss:** negative log-likelihood of the *actual scoreline* under the predicted bivariate-Poisson distribution (a proper scoring rule; trains all markets jointly). Simpler alternative: model the score grid as a flat softmax categorical + cross-entropy.
- **Validation — critical:** **strictly chronological / walk-forward** splits. Train on matches before date T, validate on a window after T, roll forward. **Never random-shuffle** — it leaks the future and inflates metrics.
- **Metrics:** Ranked Probability Score (RPS, standard for football), log-loss, Brier, and **calibration / reliability diagrams** (when we say 60%, does it happen ~60% of the time?).
- **Calibration step:** fit temperature scaling or isotonic regression on a held-out fold. Raw nets are overconfident; uncalibrated probabilities ruin value calculations.
- **Regularization (sparsity):** heavy dropout, small embeddings, weight decay, early stopping on RPS, time-decay sample weighting, optional train-on-club-football then fine-tune-on-internationals.
- **GPU usage:** model trains in minutes. Spend headroom on hyperparameter sweeps, deep ensembles (train ~20 nets, average distributions — improves calibration), and the optional per-player encoder.

---

## 8. Outputs & CLI

Per upcoming match: one forward pass → score grid → derive all markets.

Target CLI behaviour:

```
$ python predict.py --home "Argentina" --away "France"

Argentina vs France  (neutral, knockout)
  Expected goals: 1.42 - 1.18
  1X2:       Home 41.3%  Draw 26.1%  Away 32.6%
  Over 2.5:  51.8%      BTTS: 54.0%
  Top scores: 1-1 12.1% | 1-0 10.4% | 2-1 9.8%
  --- vs market (margin-removed) ---
  Market 1X2: 39.0% / 27.5% / 33.5%
  Edge: Home +2.3pp -> EV +4.1%  [value]
  Kelly (1/4): 1.0% of bankroll
```

CLI commands needed:
- **predict** a single match.
- **backtest:** replay a past tournament/season, bet only where modeled prob exceeds market-implied prob by a threshold, report ROI vs. **closing** line. Include flat-betting ROI too (so Kelly variance doesn't fool us).
- **export** probabilities to CSV.

Staking display: **fractional Kelly** only. Credible edge = profit on historical *closing* odds.

---

## 9. Open questions — discuss with me before scaffolding

1. **Score-grid size:** 0–7 each enough, or go higher for blowouts? Truncation handling?
2. **Distribution:** true bivariate Poisson vs. independent Poisson + Dixon–Coles τ correction — start with which?
3. **Form-encoder:** GRU vs. 1-layer Transformer for the initial version?
4. **Sparsity strategy:** internationals-only first, or club-pretrain + international-fine-tune from the start?
5. **Backtest target:** which historical tournament(s) to backtest on (e.g. WC 2018/2022, Euros) given odds availability?
6. **Data layer:** `polars` vs. `pandas`? Caching strategy for scraped data?
7. **Scope of v1:** do we ship value-betting evaluation on the *baselines* before any deep learning (recommended), or build the net early?

---

## 10. Tech stack & repo layout

- **Stack:** Python, PyTorch (model), polars/pandas (data), `soccerdata` (ingest), `typer` or `argparse` (CLI), matplotlib (calibration plots). No web framework.
- **Proposed layout:**

```
worldcup-model/
  data/          # ingestion + local cache
  features/      # elo, form, squad-strength, context builders
  models/        # elo.py, dixoncoles.py, xgb.py, neural.py
  eval/          # rps.py, calibration.py, backtest.py
  cli/           # predict.py, backtest.py, export.py
  README.md
  CLAUDE.md      # (this file)
```

---

## 11. Build order (weekend-scale)

1. **Phase 1:** data ingestion + cache, Elo + Dixon–Coles baselines, backtest harness, calibration plots. (Value betting is evaluable here, before any DL.)
2. **Phase 2:** neural bivariate-Poisson model + XGBoost baseline, walk-forward eval, calibration.
3. **Phase 3:** ensembling, optional per-player encoder, live odds comparison for actual WC 2026 matches.

---

## 12. Responsible note

This is a research/learning project. Gambling carries real financial risk, models rarely beat the market net of margin, and any staking output is illustrative. Keep stakes hypothetical unless you fully understand the risk.
