import unicodedata

import pandas as pd

TOP5_LEAGUES: frozenset[str] = frozenset(
    {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
)

SQUAD_FEATURES_CACHE_KEY: str = "squad_features"


def _nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name.strip().lower())


def build_squad_features(
    squads: pd.DataFrame,
    player_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Join squad rosters to club stats and aggregate to team-level features.

    Returns one row per team with columns:
        team (str)
        squad_xg_per90 (float)      minutes-weighted mean xG/90 across matched players
        squad_xa_per90 (float)      minutes-weighted mean xA/90
        squad_npxg_per90 (float)    minutes-weighted mean npxG/90
        squad_market_value (float)  placeholder 0.0 (Transfermarkt scraping deferred)
        top5_league_share (float)   fraction of squad players whose club is in TOP5_LEAGUES
        n_matched (int)             number of players successfully matched to FBref stats
        n_squad (int)               total squad size

    Matching: join squads to player_stats on player_name (case-insensitive strip).
    Log unmatched players as warnings (print to stdout).
    For unmatched players, exclude from weighted averages but count in n_squad.
    """
    squads = squads.copy()
    player_stats = player_stats.copy()

    squads["_key"] = squads["player_name"].map(_nfc)
    player_stats["_key"] = player_stats["player_name"].map(_nfc)

    stats_lookup = player_stats.set_index("_key")

    stat_cols = ["minutes", "xg_per90", "xa_per90", "npxg_per90"]
    for col in stat_cols:
        if col not in stats_lookup.columns:
            stats_lookup[col] = 0.0

    league_col: str | None = "league" if "league" in stats_lookup.columns else None

    records: list[dict] = []

    for team, group in squads.groupby("team"):
        n_squad = len(group)
        matched_rows: list[pd.Series] = []
        unmatched_names: list[str] = []

        for _, player_row in group.iterrows():
            key = player_row["_key"]
            if key in stats_lookup.index:
                stat_row = stats_lookup.loc[key]
                if isinstance(stat_row, pd.DataFrame):
                    stat_row = stat_row.iloc[0]
                matched_rows.append(stat_row)
            else:
                unmatched_names.append(player_row["player_name"])

        for name in unmatched_names:
            print(f"[squad_features] UNMATCHED player: {name!r} (team={team})")

        n_matched = len(matched_rows)

        if matched_rows:
            matched_df = pd.DataFrame(matched_rows).copy()
            weights = pd.to_numeric(matched_df["minutes"], errors="coerce").fillna(0.0)
            total_weight = weights.sum()

            def wmean(
                col: str,
                _df: pd.DataFrame = matched_df,
                _w: pd.Series = weights,
                _tw: float = total_weight,
            ) -> float:
                if col not in _df.columns or _tw == 0.0:
                    return 0.0
                vals = pd.to_numeric(_df[col], errors="coerce").fillna(0.0)
                return float((vals * _w).sum() / _tw)

            squad_xg_per90 = wmean("xg_per90")
            squad_xa_per90 = wmean("xa_per90")
            squad_npxg_per90 = wmean("npxg_per90")

            if league_col is not None:
                matched_leagues = matched_df[league_col].dropna().tolist()
            else:
                matched_leagues = []
        else:
            squad_xg_per90 = 0.0
            squad_xa_per90 = 0.0
            squad_npxg_per90 = 0.0
            matched_leagues = []

        top5_count = sum(
            1
            for _, pr in group.iterrows()
            if pr.get("club_country", "") in TOP5_LEAGUES or pr.get("club", "") in TOP5_LEAGUES
        )
        if matched_leagues:
            top5_count = sum(1 for lg in matched_leagues if lg in TOP5_LEAGUES)
            top5_league_share = top5_count / n_squad if n_squad > 0 else 0.0
        else:
            top5_league_share = top5_count / n_squad if n_squad > 0 else 0.0

        records.append(
            {
                "team": team,
                "squad_xg_per90": squad_xg_per90,
                "squad_xa_per90": squad_xa_per90,
                "squad_npxg_per90": squad_npxg_per90,
                "squad_market_value": 0.0,
                "top5_league_share": top5_league_share,
                "n_matched": n_matched,
                "n_squad": n_squad,
            }
        )

    result = pd.DataFrame(records)
    result["n_matched"] = result["n_matched"].astype(int)
    result["n_squad"] = result["n_squad"].astype(int)
    return result.reset_index(drop=True)


def get_team_squad_features(
    team: str,
    squads: pd.DataFrame,
    player_stats: pd.DataFrame,
) -> dict[str, float]:
    """Return squad feature dict for a single team. Convenience wrapper around build_squad_features."""
    all_features = build_squad_features(squads, player_stats)
    row = all_features[all_features["team"] == team]
    if row.empty:
        raise KeyError(f"Team {team!r} not found in squads DataFrame")
    return row.iloc[0].to_dict()
