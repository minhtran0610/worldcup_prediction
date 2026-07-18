"""Fetch recent WC news (BBC + The Guardian) and extract structured team form signals via Ollama.

Architecture
------------
This module is a text → structured-data transformer. It does NOT predict match
outcomes. Its output (form_score, key_absences) feeds the post-hoc λ adjustment
layer in features/llm_form_feature.py, which trims the neural model's Poisson rates.

Pipeline per team
-----------------
1. Fetch RSS feeds (BBC WC, BBC Sport, Guardian WC)  (one HTTP call each, shared)
2. Filter items by team name keywords in title + description
3. Fetch full article body for matched items  (parallel HTTP, each capped at
   _MAX_COMBINED_CHARS)
4. Concatenate (no further truncation — num_ctx is sized to fit)
5. POST to Ollama /api/chat with extraction prompt
6. Parse + validate JSON → FormAnalysis

Always fetches fresh — no cache. News updates throughout the day and the model
runs once per day before each match batch.

Fallback behaviour
------------------
Any failure (Ollama down, no articles, bad JSON) returns FormAnalysis with
form_score=0.0 and confidence=0.0. Callers apply zero λ adjustment — safe.

Ollama requirements
-------------------
- Ollama running at localhost:11434
- Model: qwen3.5:9b (recommended) or gemma4:e4b (fallback)
- think=False is critical: both models have thinking mode on by default
  and return an empty content field unless it is explicitly disabled.
"""

from __future__ import annotations

import html
import json
import os as _os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from data.ingest import guardian_api

_OLLAMA_BASE = _os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_OLLAMA_TIMEOUT = 120  # seconds — allow for cold model load (observed 42.8s generation
# alone even with a warm prompt cache; per-match firings are hours apart, so the model
# is routinely evicted from VRAM between calls and pays a full reload on top of that)
_BBC_WC_RSS = "https://feeds.bbci.co.uk/sport/football/world-cup/rss.xml"
_BBC_SPORT_RSS = "https://feeds.bbci.co.uk/sport/football/rss.xml"
_GUARDIAN_WC_RSS = "https://www.theguardian.com/football/world-cup-2026/rss"
_GUARDIAN_FOOTBALL_RSS = "https://www.theguardian.com/football/rss"
_ESPN_SOCCER_RSS = "https://www.espn.com/espn/rss/soccer/news"
_FRANCE24_SPORT_RSS = "https://www.france24.com/en/sport/rss"
_MAX_COMBINED_CHARS = 8000
_ARTICLE_FETCH_WORKERS = 8

DEFAULT_MODEL = "qwen3.5:9b"
FALLBACK_MODEL = "gemma4:e4b"

# ── Search term mapping ───────────────────────────────────────────────────────
# BBC articles use various names for national teams. Map canonical training-data
# names to the terms we search for in article titles + descriptions.
_SEARCH_TERMS: dict[str, list[str]] = {
    "Argentina": ["Argentina", "Argentine"],
    "Australia": ["Australia", "Socceroos"],
    "Austria": ["Austria"],
    "Belgium": ["Belgium", "Belgian", "Red Devils"],
    "Bosnia and Herzegovina": ["Bosnia"],
    "Brazil": ["Brazil", "Seleção"],
    "Canada": ["Canada", "Canadian", "CanMNT", "Marsch", "Davies"],
    "Cape Verde": ["Cape Verde"],
    "Colombia": ["Colombia", "Colombian"],
    "Croatia": ["Croatia", "Croatian"],
    "Czech Republic": ["Czech", "Czechia"],
    "Curaçao": ["Curaçao", "Curacao"],
    "DR Congo": ["Congo", "DR Congo"],
    "Ecuador": ["Ecuador"],
    "Egypt": ["Egypt", "Egyptian", "Salah"],
    "England": ["England", "Tuchel", "Three Lions"],
    "France": ["France", "French", "Les Bleus"],
    "Germany": ["Germany", "German"],
    "Ghana": ["Ghana"],
    "Haiti": ["Haiti"],
    "Iran": ["Iran"],
    "Iraq": ["Iraq"],
    "Ivory Coast": ["Ivory Coast", "Côte d'Ivoire"],
    "Japan": ["Japan", "Samurai Blue"],
    "Jordan": ["Jordan"],
    "Mexico": ["Mexico", "Mexican", "El Tri"],
    "Morocco": ["Morocco", "Atlas Lions"],
    "Netherlands": ["Netherlands", "Dutch", "Holland"],
    "New Zealand": ["New Zealand", "All Whites"],
    "Norway": ["Norway", "Norwegian", "Haaland"],
    "Panama": ["Panama"],
    "Paraguay": ["Paraguay"],
    "Portugal": ["Portugal", "Ronaldo"],
    "Qatar": ["Qatar"],
    "Saudi Arabia": ["Saudi Arabia", "Saudi"],
    "Scotland": ["Scotland", "Scottish"],
    "Senegal": ["Senegal"],
    "South Africa": ["South Africa", "Bafana", "Broos", "Lyle Foster"],
    "South Korea": ["South Korea", "Korea"],
    "Spain": ["Spain", "Spanish", "La Roja"],
    "Sweden": ["Sweden", "Swedish"],
    "Switzerland": ["Switzerland", "Swiss"],
    "Tunisia": ["Tunisia"],
    "Turkey": ["Turkey", "Turkish", "Türkiye"],
    "United States": ["United States", "USA", "USMNT", "US men"],
    "Uruguay": ["Uruguay"],
    "Uzbekistan": ["Uzbekistan"],
}

# ── Data class ────────────────────────────────────────────────────────────────


@dataclass
class FormAnalysis:
    """Structured form signal extracted from news text for one national team."""

    team: str
    form_score: float = 0.0  # [-1, 1]  overall narrative momentum
    key_absences: list[str] = field(default_factory=list)
    morale_signals: list[str] = field(default_factory=list)
    tactical_notes: str = ""
    performance_context: str = ""  # narrative of recent performance quality / momentum
    confidence: float = 0.0  # [0, 1]  signal richness
    sources: list[str] = field(default_factory=list)
    n_articles: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "form_score": self.form_score,
            "key_absences": self.key_absences,
            "morale_signals": self.morale_signals,
            "tactical_notes": self.tactical_notes,
            "performance_context": self.performance_context,
            "confidence": self.confidence,
            "sources": self.sources,
            "n_articles": self.n_articles,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FormAnalysis:
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})

    @classmethod
    def neutral(cls, team: str, reason: str = "") -> FormAnalysis:
        return cls(team=team, error=reason or None)


# ── RSS + article fetch ───────────────────────────────────────────────────────


def _fetch_rss(url: str) -> list[dict]:
    """Return list of {title, description, link} from an RSS feed."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            content = r.read()
    except Exception as exc:
        print(f"[llm_form] RSS fetch failed ({url}): {exc}", file=sys.stderr)
        return []
    try:
        tree = ET.fromstring(content)
    except ET.ParseError as exc:
        print(f"[llm_form] RSS parse failed: {exc}", file=sys.stderr)
        return []
    channel = tree.find("channel")
    if channel is None:
        return []
    return [
        {
            "title": i.findtext("title", ""),
            "description": i.findtext("description", ""),
            "link": i.findtext("link", "").split("?")[0],
        }
        for i in channel.findall("item")
    ]


def clean_html(text: str) -> str:
    """Strip tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


_BODY_FETCH_DOMAINS = ("bbc.", "theguardian.com", "espn.com")

# Terms that indicate nav/JS boilerplate rather than article content.
# Guardian and ESPN both embed UI snippets in <p> tags that slip past length filters.
_BOILERPLATE_TERMS = frozenset(
    {
        "close dialogue",
        "toggle caption",
        "document.addeventlistener",
        "veggieburger",
        "print subscriptions",
        "search jobs",
    }
)


def _is_boilerplate(text: str) -> bool:
    lower = text.lower()
    # ESPN video-description paragraphs always start with "play "
    if lower.startswith("play "):
        return True
    return any(term in lower for term in _BOILERPLATE_TERMS)


def _fetch_article_text(url: str) -> str:
    """Fetch full body text from a BBC, Guardian, or ESPN article. Returns '' on failure."""
    if not url or not any(d in url for d in _BODY_FETCH_DOMAINS):
        return ""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    # Collect <p> tags with enough content (≥60 chars after tag-stripping)
    paras = re.findall(r"<p[^>]*>(.*?)</p>", raw, re.DOTALL)
    cleaned = [clean_html(p) for p in paras]
    cleaned = [
        p
        for p in cleaned
        if len(p) >= 60
        and not p.startswith("BBC")
        and "cookie" not in p.lower()
        and "subscribe" not in p.lower()
        and not _is_boilerplate(p)
    ]
    return " ".join(cleaned)[:_MAX_COMBINED_CHARS]


def fetch_match_news(home: str, away: str, max_articles: int = 12) -> tuple[list[str], list[str]]:
    """Fetch articles for both teams, interleaved to guarantee equal representation.

    Fetches up to max_articles//2 per team independently (title-first ordering),
    then interleaves home/away articles so neither team's content is displaced by
    the other when the combined text is truncated. Deduplicates by URL.
    Returns (texts, urls).
    """
    per_team = max(2, max_articles // 2)
    home_texts, home_urls = fetch_team_news(home, max_articles=per_team)
    away_texts, away_urls = fetch_team_news(away, max_articles=per_team)

    seen: set[str] = set()
    texts: list[str] = []
    urls: list[str] = []
    for h_text, h_url, a_text, a_url in zip(
        home_texts + [None] * per_team,
        home_urls + [None] * per_team,
        away_texts + [None] * per_team,
        away_urls + [None] * per_team,
        strict=False,
    ):
        for text, url in ((h_text, h_url), (a_text, a_url)):
            if text and url and url not in seen:
                seen.add(url)
                texts.append(text)
                urls.append(url)

    return texts[:max_articles], urls[:max_articles]


def fetch_team_news(team: str, max_articles: int = 6) -> tuple[list[str], list[str]]:
    """Return (texts, urls) — article bodies + source URLs for *team*.

    Searches BBC, Guardian, ESPN, and France24 RSS feeds.
    Filters by case-insensitive keyword match in title or description.
    Fetches article bodies in parallel for BBC, Guardian, and ESPN.
    France24 uses RSS descriptions (clean, sufficient length).

    Returns ([], []) if no articles found or on network failure.
    """
    search_terms = _SEARCH_TERMS.get(team, [team])
    search_lower = [s.lower() for s in search_terms]

    # WC-specific feeds first; broader feeds as fallback.
    # ESPN and France24 provide form-focused match reports missing from BBC/Guardian.
    all_items: list[dict] = []
    for rss_url in [
        _BBC_WC_RSS,
        _GUARDIAN_WC_RSS,
        _ESPN_SOCCER_RSS,
        _FRANCE24_SPORT_RSS,
        _BBC_SPORT_RSS,
        _GUARDIAN_FOOTBALL_RSS,
    ]:
        all_items.extend(_fetch_rss(rss_url))

    # Deduplicate by URL and content fingerprint (same article, different URLs).
    # Separate title matches (team is primary subject) from description-only
    # matches where the term must appear at least twice (filters host-country noise).
    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()
    title_matches: list[dict] = []
    desc_matches: list[dict] = []
    for item in all_items:
        link = item["link"]
        if not link or link in seen_urls:
            continue
        # Content fingerprint: first 120 chars of title+description
        fingerprint = (item["title"] + item["description"])[:120].lower()
        if fingerprint in seen_fingerprints:
            continue
        title_lower = item["title"].lower()
        # Strip HTML before counting — Guardian descriptions embed team names in href attrs
        desc_clean = clean_html(item["description"]).lower()
        in_title = any(term in title_lower for term in search_lower)
        # Description-only: require term appears at least twice in plain text to avoid noise
        if not in_title:
            desc_count = sum(desc_clean.count(term) for term in search_lower)
            in_desc = desc_count >= 2
        else:
            in_desc = False
        if in_title or in_desc:
            seen_urls.add(link)
            seen_fingerprints.add(fingerprint)
            (title_matches if in_title else desc_matches).append(item)

    # Title matches first — these are articles where the team is the primary subject
    matched = (title_matches + desc_matches)[:max_articles]

    if not matched:
        return [], []

    urls = [m["link"] for m in matched]
    rss_descs = {m["link"]: clean_html(m["title"] + ". " + m["description"]) for m in matched}

    # BBC, Guardian, and ESPN: fetch full article body (<p> extraction works for all three).
    # Guardian and ESPN boilerplate is filtered in _fetch_article_text via _is_boilerplate().
    # France24 and other sources: RSS description is clean and sufficient.
    body_fetch_urls = [u for u in urls if any(d in u for d in _BODY_FETCH_DOMAINS)]
    rss_only_urls = [u for u in urls if not any(d in u for d in _BODY_FETCH_DOMAINS)]

    texts: list[str] = []

    if body_fetch_urls:
        with ThreadPoolExecutor(
            max_workers=min(_ARTICLE_FETCH_WORKERS, len(body_fetch_urls))
        ) as exe:
            future_to_url = {exe.submit(_fetch_article_text, url): url for url in body_fetch_urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                body = future.result()
                # Fall back to RSS description if body extraction yielded too little
                texts.append(body if len(body) >= 150 else rss_descs.get(url, ""))

    for url in rss_only_urls:
        desc = rss_descs.get(url, "")
        if desc:
            texts.append(desc)

    texts = [t for t in texts if t]
    return texts, urls


def fetch_team_match_report(
    team: str,
    opponent: str,
    match_date: pd.Timestamp,
    api_key: str | None = None,
) -> tuple[list[str], list[str]]:
    """Fetch a post-match report for one specific completed match via the
    Guardian Content API's archive search.

    fetch_team_news (RSS-based) can't reach matches whose reports have
    already rotated out of the live feed — this function exists for exactly
    that case: retroactively recovering earlier WC2026 match reports.

    Searches a 2-day window starting on match_date. Filters results to those
    whose title mentions the team (via the existing _SEARCH_TERMS keyword
    map) to reduce false positives from the Guardian's broader query match.
    Returns ([], []) on no results or API failure.
    """
    query = f"{team} {opponent}"
    to_date = cast(pd.Timestamp, match_date + pd.Timedelta(days=2))
    articles = guardian_api.search_articles(query, match_date, to_date, api_key=api_key)

    search_terms = [t.lower() for t in _SEARCH_TERMS.get(team, [team])]
    texts: list[str] = []
    urls: list[str] = []
    for a in articles:
        title_lower = a["title"].lower()
        if not any(term in title_lower for term in search_terms):
            continue
        body = clean_html(a["body_html"])[:_MAX_COMBINED_CHARS]
        if body:
            texts.append(body)
            urls.append(a["url"])
    return texts, urls


# ── Ollama extraction ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a football analyst extracting a team's CURRENT performance state from recent "
    "news text. Extract ANYTHING the article says about how the team is playing — results, "
    "scorelines, style, morale, momentum, manager situation, key players. Be generous: "
    "if the article mentions it in the context of this team's performance or tournament "
    "journey, include it.\n\n"
    "Extract especially:\n"
    "  - Results and how they happened: scorelines, margin of victory/defeat, whether the "
    "result flattered or understated performance ('thumping', 'comfortable', 'lucky', "
    "'dominant', 'outplayed', 'deserved more').\n"
    "  - Momentum and trajectory: improving or declining, growing confidence or pressure, "
    "knockout-stage readiness, historic achievements that affect belief.\n"
    "  - Team and manager signals: tactical setup, dressing-room atmosphere, coach "
    "statements, player form, squad cohesion or unrest.\n"
    "  - Confirmed absences: players explicitly described as OUT, injured, suspended, "
    "or sent off FOR THE NEXT MATCH. Do NOT include players described as 'doubtful', "
    "'carrying a knock', 'returned to training', 'returning', 'expected back', 'set to "
    "return', 'available', or 'back in the squad' — these players are NOT absent. "
    "Past absences that have ended are NOT absences.\n\n"
    "RULES:\n"
    "1. Only use information EXPLICITLY stated in the text — no external knowledge.\n"
    "2. Round names are precise, do not blur them: a team only played 'the final' if that "
    "match decided the tournament champion. A semi-final loser did NOT play 'in the final' "
    "— they lost the semi-final and now play the third-place / bronze medal match. Sports "
    "journalism often nicknames the third-place game 'the final' or 'the bronze final' in "
    "headlines — do not let that nickname make you call an actual semi-final result 'the "
    "final'. If text is ambiguous about which round a result belongs to, describe the result "
    "without naming the round rather than guessing.\n"
    "3. key_absences entries are bare player names ONLY — never append parentheticals, "
    "performance commentary, or fitness caveats to a name (e.g. write 'Player X', not "
    "'Player X (fit again but underperforming)'). If a player's absence status carries "
    "nuance worth capturing, put that nuance in performance_context instead, and only add "
    "the name to key_absences if they are cleanly confirmed absent.\n"
    "4. performance_context: ALWAYS fill this. Summarise what the articles say about this "
    "team's form and momentum. If genuinely nothing is said, explain what the articles "
    "covered instead.\n"
    "5. form_score [-1, 1]: overall momentum signal. Positive = playing well / rising / "
    "confident. Negative = struggling / unsettled / weakened. 0 = nothing to go on.\n"
    "6. confidence [0, 1]: driven primarily by SOURCE COUNT, not how decisively any single "
    "source is worded — one forcefully-worded article is still one data point. 0 = team "
    "barely mentioned. 0.15-0.35 = one article with a clear but singular statement. "
    "0.4-0.6 = one detailed article, or two loosely agreeing. 0.65-0.8 = two to three "
    "articles independently corroborating the same direction. 0.85-1.0 = reserved for four "
    "or more corroborating articles, or a detailed match report independently confirmed "
    "elsewhere. Most inputs should land in the middle of this range — treat 0.9+ as rare, "
    "not a default for 'the article sounded confident.'\n"
    "7. Output valid JSON only — no markdown, no text outside the JSON object."
)

_USER_TEMPLATE = (
    "Analyse the following recent news text about {team} and extract structured form signals.\n\n"
    "CRITICAL: You are ONLY extracting data for {team}. If the article covers a match between "
    "{team} and an opponent, ignore the opponent entirely — do not list any opponent players "
    "in key_absences, do not score the opponent's morale, do not mix up the two sides.\n\n"
    "CRITICAL: Get round names right. A semi-final loss is NOT 'the final', even if the text "
    "nicknames the third-place playoff 'the final' or 'the bronze final' — never repeat that "
    "nickname as if it were the actual final.\n\n"
    "{stage_context}"
    "TEXT:\n{text}\n\n"
    "Return JSON with EXACTLY this schema (no extra keys):\n"
    '{{"form_score": <float -1.0 to 1.0: overall narrative sentiment for {team} — '
    "factor in performance quality, momentum, morale, and absences together>, "
    '"performance_context": "<string: REQUIRED — 1-3 sentences on {team}\'s current form '
    "narrative (performance quality, momentum, morale, press perception) OR, if no signal, "
    'a brief explanation of what the articles covered and why no form signal could be extracted>", '
    '"key_absences": [<strings: BARE player names only, confirmed absent for {team}\'s NEXT '
    "match — no parentheticals or commentary appended to a name (fitness/performance nuance "
    "goes in performance_context instead); never list players from the opposing team>], "
    '"morale_signals": [<strings: direct short phrases from text indicating {team} morale or '
    "team atmosphere>], "
    '"tactical_notes": "<string: {team} tactical changes or coach statements, empty if none>", '
    '"confidence": <float 0.0 to 1.0: how strongly the text supports your form_score. '
    "This value scales how much your judgement moves the prediction, so use the full range "
    "— do not round to the nearest anchor below, use the exact value the evidence justifies. "
    "Driven primarily by SOURCE COUNT, not how decisively any one source is worded: "
    "0=team barely mentioned or no clear performance signal; 0.15-0.35=one article with a "
    "clear but singular statement; 0.4-0.6=one detailed article, or two loosely agreeing; "
    "0.65-0.8=two to three articles independently corroborating the same direction; "
    "0.85-1.0=reserved for four or more corroborating articles, or a detailed match report "
    "independently confirmed elsewhere. Most inputs should land in the middle of this range "
    "— treat 0.9+ as rare, not a default for 'the article sounded confident.'>}}"
)

# ── Trajectory prompt (multi-match, chronologically ordered) ──────────────────
# Separate from _SYSTEM_PROMPT/_USER_TEMPLATE above, which stay single-article
# and are still used by the pre-match sentiment path. This prompt explicitly
# frames the input as an ordered sequence of post-match reports and asks the
# model to reason about how form has changed ACROSS matches, not just what is
# true in any one of them — see data.ingest.trajectory.get_team_trajectory.

# The trajectory prompt asks for a longer response than the single-match one
# (2-4 sentences of cross-match narrative vs. 1-3, plus lists potentially
# aggregated across multiple matches) — the default 400-token budget can cut
# the JSON off mid-generation on richer inputs, which surfaces as a parse
# failure rather than a truncation warning. Bumped from 600: a 7-match,
# heavily-eventful trajectory (e.g. a deep-tournament team) can produce a
# performance_context alone that eats most of a 600-token budget, leaving the
# schema's LAST field at risk of truncation — confidence was moved to be the
# second field for this reason (see _USER_TEMPLATE_TRAJECTORY), but the extra
# headroom here removes the pressure entirely rather than just relocating it.
_TRAJECTORY_NUM_PREDICT = 900

# Ollama defaults to a 4096-token context window regardless of what a model
# architecturally supports, UNLESS num_ctx is set explicitly in the request —
# verified empirically (a 52k-char prompt with a marker near the end came
# back with prompt_eval_count=4096 and the model never saw the marker). Our
# previous _MAX_TRAJECTORY_CHARS=16000 app-level cap was masking this same
# ceiling, and a naive join-then-slice was silently dropping whichever
# matches didn't fit — e.g. a 5-match trajectory where the first two
# matches' articles alone exceeded the cap, so the model never saw the last
# three matches (including a 3-0 win) at all. The single-match sentiment
# path's combined-articles slice had the same bug (fewer articles, but still
# capable of dropping later ones once several per team exceeded the cap) —
# removed there too.
#
# Empirically tested on this project's RTX 4070 Ti Super (16GB VRAM) via
# `ollama ps` + `nvidia-smi` while loading qwen3.5:9b (Q4_K_M):
#   num_ctx=65536   -> 100% GPU, ~11GB used, ~5.4GB free
#   num_ctx=131072  -> 100% GPU, ~14GB used, ~2.6GB free  (largest still 100% GPU)
#   num_ctx=262144  -> model's advertised max, but spills to 31%/69% CPU/GPU
#                      (much slower — not worth it)
# 131072 chosen over 65536: a deep-tournament team (finalist, 7 matches) with
# heavy Guardian coverage could plausibly need ~70k+ tokens, uncomfortably
# close to 65536's ceiling — 131072 gives real margin for that actual worst
# case while still running 100% on GPU. The per-match/combined character
# truncation this replaced is removed entirely — num_ctx is now the real,
# properly-sized boundary instead of an arbitrary undersized guess.
#
# Shared by both the trajectory and single-match sentiment paths rather than
# giving sentiment its own smaller value: Ollama keys its loaded model
# instance by (model, options), so a differing num_ctx between the two call
# sites forces a model reload every time the pipeline alternates between
# them. One shared value keeps a single instance loaded for both.
_NUM_CTX = 131072

_SYSTEM_PROMPT_TRAJECTORY = (
    "You are a football analyst extracting a team's CURRENT performance trajectory from a "
    "sequence of post-match reports, one per completed match, given to you in chronological "
    "order. Extract ANYTHING the reports say about how the team is playing — results, "
    "scorelines, style, morale, momentum, manager situation, key players — and reason about "
    "how it has CHANGED across matches, not just what is true in any single one.\n\n"
    "Extract especially:\n"
    "  - Results and how they happened: scorelines, margin of victory/defeat, whether the "
    "result flattered or understated performance ('thumping', 'comfortable', 'lucky', "
    "'dominant', 'outplayed', 'deserved more').\n"
    "  - Trajectory: is the team improving, declining, or inconsistent across these matches? "
    "Weight more recent matches more heavily than earlier ones when they conflict — a team "
    "that lost game 1 and dominated games 2-3 is trending up, not down.\n"
    "  - Team and manager signals: tactical setup, dressing-room atmosphere, coach "
    "statements, player form, squad cohesion or unrest, and whether these are getting "
    "better or worse across the sequence.\n"
    "  - Confirmed absences: players explicitly described as OUT, injured, suspended, "
    "or sent off FOR THE NEXT MATCH. Do NOT include players described as 'doubtful', "
    "'carrying a knock', 'returned to training', 'returning', 'expected back', 'set to "
    "return', 'available', or 'back in the squad' — these players are NOT absent. "
    "Past absences that have ended are NOT absences.\n\n"
    "RULES:\n"
    "1. Only use information EXPLICITLY stated in the text — no external knowledge.\n"
    "2. Each match block's header (=== Match N of M — vs OPPONENT (date), ROUND, ... ===) "
    "states the ACTUAL round for that match — copy that round name VERBATIM whenever you "
    "refer to the round, do not paraphrase or recompute it. WARNING: this is the 2026 "
    "48-team World Cup, which has FIVE knockout rounds — Round of 32, Round of 16, "
    "Quarter-final, Semi-final, Final — one more than the 32-team format's traditional four "
    "(Round of 16, QF, SF, Final) that you may expect from historical World Cups. Do NOT "
    "infer a match's round by counting backwards from the final assuming only four knockout "
    "rounds exist — that undercounts by one and mislabels every round from Round of 16 "
    "onward. The header string is the only source of truth for round names; your own sense "
    "of 'how many rounds a World Cup has' is wrong for this tournament and must be ignored. "
    "Also, never let the article prose override the header: journalism often nicknames "
    "matches loosely (e.g. calling the third-place playoff 'the final' or 'the bronze final' "
    "in a headline) — those nicknames describe a DIFFERENT match than the one in that "
    "block's header, so ignore them when labeling this block's round.\n"
    "3. performance_context: ALWAYS fill this. Summarise the team's trajectory across ALL "
    "matches given, not just the most recent one — name the direction (improving/declining/"
    "steady) explicitly in prose. If genuinely nothing is said, explain what the reports "
    "covered instead.\n"
    "4. form_score [-1, 1]: overall CURRENT momentum, informed by the trajectory across all "
    "matches but weighted toward the most recent. Positive = playing well / rising / "
    "confident. Negative = struggling / unsettled / weakened. 0 = nothing to go on.\n"
    "5. confidence [0, 1]: driven primarily by how many DISTINCT matches carry usable signal, "
    "not how decisively any single report is worded, and NOT by whether the trajectory itself "
    "is steady. A team that swings dominant-shaky-dominant-collapsed across matches can still "
    "earn top confidence if each swing is clearly documented — confidence measures evidence "
    "coverage, never narrative tidiness; do not discount it for a volatile or contradictory "
    "trajectory, that IS the trajectory. 0 = team barely mentioned in any report. 0.15-0.35 = "
    "one match with a clear but singular statement. 0.4-0.6 = one match covered in detail, or "
    "two matches with thin/loosely documented signal. 0.65-0.8 = clear, usable signal in two "
    "to three matches, regardless of whether they agree in direction. 0.85-1.0 = reserved for "
    "clear, usable signal in four or more matches, regardless of whether the trajectory across "
    "them is steady or volatile. Most inputs should land in the middle of this range — treat "
    "0.9+ as rare, not a default for 'the reports sounded confident.'\n"
    "6. Output valid JSON only — no markdown, no text outside the JSON object."
)

_USER_TEMPLATE_TRAJECTORY = (
    "Analyse the following chronological sequence of post-match reports about {team} and "
    "extract structured trajectory signals.\n\n"
    "CRITICAL: You are ONLY extracting data for {team}. Ignore opponent-specific detail "
    "except as context for {team}'s own performance — do not list opponent players in "
    "key_absences, do not score the opponent's morale.\n\n"
    "CRITICAL: Each block header below states that match's actual round — copy it VERBATIM "
    "when naming the round, do not recompute it. This is the 2026 48-team World Cup: FIVE "
    "knockout rounds exist (Round of 32, Round of 16, Quarter-final, Semi-final, Final), one "
    "more than older 32-team World Cups' four — do not assume the traditional four-round "
    "structure and count backwards from the final, that mislabels every round from Round of "
    "16 onward by one. Also ignore any different round name or nickname (e.g. 'the final', "
    "'the bronze final') that shows up in the article text itself, since that nickname "
    "likely describes a different match than the one in that block.\n\n"
    "REPORTS (chronological order, earliest match first):\n{text}\n\n"
    "Return JSON with EXACTLY this schema (no extra keys) — form_score and confidence come "
    "FIRST, before the free-text fields, so they are never at risk of getting cut off if your "
    "response runs long:\n"
    '{{"form_score": <float -1.0 to 1.0: {team}\'s CURRENT momentum after all matches above, '
    "weighted toward the most recent>, "
    '"confidence": <float 0.0 to 1.0: how strongly the reports collectively support your '
    "form_score. Use the full range — do not round to the nearest anchor below, use the "
    "exact value the evidence justifies. Driven primarily by how many DISTINCT matches carry "
    "usable signal, not how decisively any single report is worded, and NOT by whether the "
    "trajectory itself is steady — a team that swings dominant-shaky-dominant-collapsed can "
    "still earn top confidence if each swing is clearly documented; confidence measures "
    "evidence coverage, never narrative tidiness. 0=team barely mentioned in any report; "
    "0.15-0.35=one match with a clear but singular statement; 0.4-0.6=one match covered in "
    "detail, or two matches with thin/loosely documented signal; 0.65-0.8=clear, usable "
    "signal in two to three matches, regardless of whether they agree in direction; 0.85-1.0="
    "reserved for clear, usable signal in four or more matches, regardless of whether the "
    "trajectory across them is steady or volatile. Most inputs should land in the middle of "
    "this range — treat 0.9+ as rare, not a default for 'the reports sounded confident.'>, "
    '"performance_context": "<string: REQUIRED — 2-4 sentences describing how {team}\'s form '
    "has evolved across these matches (improving/declining/steady) and why, OR if no signal, "
    'a brief explanation of what the reports covered and why no trajectory could be extracted>", '
    '"key_absences": [<strings: names of {team} players confirmed absent for their NEXT match — '
    "never list players from opposing teams>], "
    '"morale_signals": [<strings: direct short phrases from the reports indicating {team} '
    "morale or atmosphere, across all matches>], "
    '"tactical_notes": "<string: {team} tactical changes or coach statements across the '
    'reports, empty if none>"}}'
)


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_markdown_fence(content: str) -> str:
    """Strip a wrapping ```json ... ``` (or plain ``` ... ```) code fence.

    `format: "json"` isn't always strictly enforced — qwen3.5:9b has been
    observed wrapping otherwise-valid JSON in a markdown fence despite it,
    which breaks json.loads at char 0 (a backtick, not `{`). Returns content
    unchanged if it isn't fenced.
    """
    match = _CODE_FENCE_RE.match(content.strip())
    return match.group(1) if match else content


def _call_ollama(
    system_prompt: str,
    user_content: str,
    model: str,
    num_predict: int = 400,
    num_ctx: int = _NUM_CTX,
) -> dict:
    """POST extraction request to Ollama. Returns parsed JSON dict.

    num_predict caps the output token budget. The default (400) fits the
    single-match prompt's shorter expected output; callers asking for a
    longer response (e.g. analyse_team_trajectory's multi-match synthesis)
    should pass a larger value — otherwise the model's JSON can get cut off
    mid-generation, which surfaces as a JSON parse failure here, not a
    truncation warning.

    num_ctx caps the total context window (prompt + response). Ollama
    silently truncates to a small default (observed: 4096 tokens) if this
    isn't set explicitly, regardless of what the model itself supports —
    always pass a value sized for the actual input, not just accept the
    default.

    Raises ConnectionError if Ollama is not reachable.
    Raises ValueError if response is not valid JSON.
    """
    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "think": False,  # critical: disables internal reasoning trace
            "format": "json",
            "options": {
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "temperature": 0.1,
                "top_k": 20,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
    ).encode()

    req = urllib.request.Request(
        f"{_OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_OLLAMA_TIMEOUT) as r:
            result = json.loads(r.read())
    except OSError as exc:
        raise ConnectionError(f"Ollama not reachable at {_OLLAMA_BASE}: {exc}") from exc

    content = result.get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Empty content from Ollama (model={model}). Check think=False.")

    try:
        return json.loads(_strip_markdown_fence(content))
    except json.JSONDecodeError as exc:
        # done_reason="length" means num_predict cut generation off mid-JSON;
        # anything else (e.g. "stop") means the model finished normally but
        # produced non-JSON content (a leaked <think> block, prose preamble,
        # etc.) — two different root causes needing two different fixes, so
        # surface the raw content rather than just "invalid JSON" every time.
        done_reason = result.get("done_reason", "unknown")
        raise ValueError(
            f"Ollama returned unparseable JSON (model={model}, done_reason={done_reason!r}, "
            f"content_len={len(content)}): {content[:800]!r}"
        ) from exc


def _validate_raw(raw: dict, team: str) -> FormAnalysis:
    """Coerce and clamp raw LLM JSON into a FormAnalysis."""
    try:
        form_score = float(raw.get("form_score", 0.0))
        form_score = max(-1.0, min(1.0, form_score))
    except (TypeError, ValueError):
        form_score = 0.0

    try:
        confidence = float(raw.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    absences = [str(p) for p in raw.get("key_absences", []) if p]
    signals = [str(s) for s in raw.get("morale_signals", []) if s]
    notes = str(raw.get("tactical_notes", "")).strip()
    perf_ctx = str(raw.get("performance_context", "")).strip()

    return FormAnalysis(
        team=team,
        form_score=form_score,
        key_absences=absences,
        morale_signals=signals,
        tactical_notes=notes,
        performance_context=perf_ctx,
        confidence=confidence,
    )


def analyse_team_form(
    team: str,
    texts: list[str],
    urls: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    stage: str | None = None,
) -> FormAnalysis:
    """Extract structured form signals from article texts using Ollama.

    stage, if given, is the team's own upcoming-fixture round (e.g.
    "Match for third place") from the live schedule — injected as a grounding
    line so the model doesn't have to guess the CURRENT fixture's round from
    ambiguous article prose (journalists nickname the third-place playoff
    "the final" often enough to cause real mislabeling — see
    data.ingest.trajectory's per-match stage header for the equivalent fix on
    the trajectory path, which grounds PAST match rounds instead).

    Returns FormAnalysis.neutral() immediately if texts is empty (no LLM call).
    Returns neutral FormAnalysis with error set on any Ollama/parse failure.
    """
    if not texts:
        return FormAnalysis.neutral(team)

    # No character truncation on the joined whole — see the _NUM_CTX comment.
    # Each article is already capped at _MAX_COMBINED_CHARS chars individually
    # (in _fetch_article_text / fetch_team_match_report), so this only joins
    # already-bounded pieces instead of re-slicing across all of them and
    # risking silently dropping later articles the way the old cap did.
    combined = "\n\n---\n\n".join(texts)

    stage_context = (
        f"CONTEXT: {team}'s next scheduled match is the {stage}. If the news below "
        f"describes an upcoming fixture for {team}, that fixture IS the {stage} — do not "
        f"call it 'the final' or any other round unless {stage!r} is literally 'Final'.\n\n"
        if stage
        else ""
    )

    try:
        raw = _call_ollama(
            _SYSTEM_PROMPT,
            _USER_TEMPLATE.format(team=team, text=combined, stage_context=stage_context),
            model,
            num_ctx=_NUM_CTX,
        )
    except ConnectionError as exc:
        print(f"[llm_form] Ollama unavailable: {exc}", file=sys.stderr)
        return FormAnalysis.neutral(team, str(exc))
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[llm_form] JSON parse failed for {team!r}: {exc}", file=sys.stderr)
        return FormAnalysis.neutral(team, str(exc))
    except Exception as exc:
        print(f"[llm_form] Unexpected error for {team!r}: {exc}", file=sys.stderr)
        return FormAnalysis.neutral(team, str(exc))

    analysis = _validate_raw(raw, team)
    analysis.sources = urls or []
    analysis.n_articles = len(texts)
    return analysis


def analyse_team_trajectory(
    team: str,
    match_blocks: list[str],
    urls: list[str] | None = None,
    model: str = DEFAULT_MODEL,
) -> FormAnalysis:
    """Extract a single trajectory-level FormAnalysis from N pre-labeled,
    chronologically-ordered post-match report blocks (see
    data.ingest.trajectory.get_team_trajectory for block construction —
    each block covers one completed match, labeled with match number,
    opponent, and result).

    Structurally parallel to analyse_team_form, but uses a prompt
    (_SYSTEM_PROMPT_TRAJECTORY / _USER_TEMPLATE_TRAJECTORY) that explicitly
    frames the input as an ordered sequence and asks the model to reason
    about how form has changed across matches — one LLM call per team,
    rather than one per match aggregated numerically afterward.

    Returns FormAnalysis.neutral() immediately if match_blocks is empty (no
    LLM call). Returns neutral FormAnalysis with error set on any
    Ollama/parse failure. n_articles is set from len(urls) — one URL per
    source article across all matches, not per match.
    """
    if not match_blocks:
        return FormAnalysis.neutral(team)

    # No character truncation here — num_ctx=_NUM_CTX (see comment above the
    # constant) is sized to comfortably fit every match's full article text,
    # so there's no need to cut anything and risk silently dropping a match
    # the way the old char-cap-on-the-joined-whole did.
    combined = "\n\n---\n\n".join(match_blocks)

    try:
        raw = _call_ollama(
            _SYSTEM_PROMPT_TRAJECTORY,
            _USER_TEMPLATE_TRAJECTORY.format(team=team, text=combined),
            model,
            num_predict=_TRAJECTORY_NUM_PREDICT,
            num_ctx=_NUM_CTX,
        )
    except ConnectionError as exc:
        print(f"[llm_form] Ollama unavailable: {exc}", file=sys.stderr)
        return FormAnalysis.neutral(team, str(exc))
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[llm_form] JSON parse failed for {team!r}: {exc}", file=sys.stderr)
        return FormAnalysis.neutral(team, str(exc))
    except Exception as exc:
        print(f"[llm_form] Unexpected error for {team!r}: {exc}", file=sys.stderr)
        return FormAnalysis.neutral(team, str(exc))

    analysis = _validate_raw(raw, team)
    analysis.sources = urls or []
    analysis.n_articles = len(urls or [])
    return analysis


# ── High-level entry point ────────────────────────────────────────────────────


def get_team_form_analysis(
    team: str,
    model: str = DEFAULT_MODEL,
    max_articles: int = 6,
) -> FormAnalysis:
    """Fetch news and run LLM extraction for one team.

    Always fetches fresh — no cache. News changes throughout the day.
    Always returns a valid FormAnalysis — never raises. Degrades to
    form_score=0.0 / confidence=0.0 on any failure.
    """
    print(f"[llm_form] Fetching news for {team!r}...", file=sys.stderr)
    texts, urls = fetch_team_news(team, max_articles=max_articles)

    if not texts:
        print(f"[llm_form] No articles found for {team!r} — returning neutral.", file=sys.stderr)
        return FormAnalysis.neutral(team)

    print(
        f"[llm_form] {team!r}: {len(texts)} article(s) → running {model}...",
        file=sys.stderr,
    )
    analysis = analyse_team_form(team, texts, urls=urls, model=model)
    analysis.n_articles = len(texts)
    return analysis


def get_match_form_analysis(
    home: str,
    away: str,
    model: str = DEFAULT_MODEL,
    max_articles: int = 8,
    stage: str | None = None,
) -> dict[str, FormAnalysis]:
    """Fetch articles for both teams combined, extract signals for each in one LLM call.

    stage, if given, is this fixture's own round (e.g. "Match for third place") —
    both teams share it, since it's the same match. See analyse_team_form for why
    this grounding matters.

    Returns {home: FormAnalysis, away: FormAnalysis}.
    A match preview article contributes signal to both sides simultaneously.
    """
    print(f"[llm_form] Fetching match news for {home!r} vs {away!r}...", file=sys.stderr)
    per_team = max(2, max_articles // 2)
    home_texts, home_urls = fetch_team_news(home, max_articles=per_team)
    away_texts, away_urls = fetch_team_news(away, max_articles=per_team)

    if not home_texts and not away_texts:
        print(
            f"[llm_form] No articles found for {home} vs {away} — returning neutral.",
            file=sys.stderr,
        )
        return {home: FormAnalysis.neutral(home), away: FormAnalysis.neutral(away)}

    print(
        f"[llm_form] {home}: {len(home_texts)} article(s), "
        f"{away}: {len(away_texts)} article(s) → running {model}...",
        file=sys.stderr,
    )
    home_fa = analyse_team_form(home, home_texts, urls=home_urls, model=model, stage=stage)
    away_fa = analyse_team_form(away, away_texts, urls=away_urls, model=model, stage=stage)
    return {home: home_fa, away: away_fa}


def get_all_teams_form(
    teams: list[str],
    model: str = DEFAULT_MODEL,
) -> dict[str, FormAnalysis]:
    """Run form analysis for a list of teams, returning {team: FormAnalysis}.

    Teams are processed sequentially (Ollama serialises GPU inference anyway).
    HTTP article fetches within each team are parallelised.
    """
    return {team: get_team_form_analysis(team, model=model) for team in teams}


def get_all_matches_form(
    match_pairs: list[tuple[str, str, str | None]],
    model: str = DEFAULT_MODEL,
) -> dict[str, FormAnalysis]:
    """Run combined match-level form analysis for a list of (home, away, stage) triples.

    stage is that fixture's own round (e.g. "Match for third place"), or None if
    unknown — forwarded to get_match_form_analysis for round-name grounding.

    Each triple is one LLM call reading articles for both teams simultaneously.
    Returns {team_name: FormAnalysis} for all teams across all pairs.
    """
    results: dict[str, FormAnalysis] = {}
    for home, away, stage in match_pairs:
        pair_results = get_match_form_analysis(home, away, model=model, stage=stage)
        results.update(pair_results)
    return results
