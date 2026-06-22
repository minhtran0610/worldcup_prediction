"""Fetch recent BBC Sport WC news and extract structured team form signals via Ollama.

Architecture
------------
This module is a text → structured-data transformer. It does NOT predict match
outcomes. Its output (form_score, key_absences) feeds the post-hoc λ adjustment
layer in features/llm_form_feature.py, which trims the neural model's Poisson rates.

Pipeline per team
-----------------
1. Fetch BBC WC RSS feed  (one HTTP call, shared across all teams)
2. Filter items by team name keywords in title + description
3. Fetch full article body for matched items  (parallel HTTP)
4. Concatenate + truncate to 2500 chars
5. POST to Ollama /api/chat with extraction prompt
6. Parse + validate JSON → FormAnalysis
7. Cache to data/raw/llm_form_{team}_{YYYY-MM-DD}.json

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
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_OLLAMA_BASE = "http://localhost:11434"
_OLLAMA_TIMEOUT = 45  # seconds — allow for cold model load
_CACHE_DIR = Path("data/raw")
_BBC_WC_RSS = "https://feeds.bbci.co.uk/sport/football/world-cup/rss.xml"
_BBC_SPORT_RSS = "https://feeds.bbci.co.uk/sport/football/rss.xml"
_MAX_COMBINED_CHARS = 2500
_ARTICLE_FETCH_WORKERS = 4

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
    "Canada": ["Canada", "Canadian"],
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
    "South Africa": ["South Africa", "Bafana"],
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


# ── Cache helpers ─────────────────────────────────────────────────────────────


def _today_cache_path(team: str) -> Path:
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    safe = re.sub(r"[^\w]", "_", team.lower().strip())
    return _CACHE_DIR / f"llm_form_{safe}_{today}.json"


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


def _clean_html(text: str) -> str:
    """Strip tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_article_text(url: str) -> str:
    """Fetch full body text from a BBC Sport article. Returns '' on failure."""
    if not url or "bbc." not in url:
        return ""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    # Collect <p> tags with enough content (≥60 chars after tag-stripping)
    paras = re.findall(r"<p[^>]*>(.*?)</p>", raw, re.DOTALL)
    cleaned = [_clean_html(p) for p in paras]
    cleaned = [
        p for p in cleaned if len(p) >= 60 and not p.startswith("BBC") and "cookie" not in p.lower()
    ]
    return " ".join(cleaned)[:_MAX_COMBINED_CHARS]


def fetch_team_news(team: str, max_articles: int = 3) -> tuple[list[str], list[str]]:
    """Return (texts, urls) — article bodies + source URLs for *team*.

    Searches BBC WC RSS (primary) and BBC Sport RSS (secondary).
    Filters by case-insensitive keyword match in title or description.
    Fetches article bodies in parallel.

    Returns ([], []) if no articles found or on network failure.
    """
    search_terms = _SEARCH_TERMS.get(team, [team])
    search_lower = [s.lower() for s in search_terms]

    # Fetch both RSS feeds (WC-specific is more relevant)
    all_items: list[dict] = []
    for rss_url in [_BBC_WC_RSS, _BBC_SPORT_RSS]:
        all_items.extend(_fetch_rss(rss_url))

    # Deduplicate by URL and filter to team-relevant items
    seen: set[str] = set()
    matched: list[dict] = []
    for item in all_items:
        link = item["link"]
        if not link or link in seen:
            continue
        haystack = (item["title"] + " " + item["description"]).lower()
        if any(term in haystack for term in search_lower):
            seen.add(link)
            matched.append(item)
        if len(matched) >= max_articles:
            break

    if not matched:
        return [], []

    urls = [m["link"] for m in matched]

    # Fetch article bodies in parallel
    texts: list[str] = []
    with ThreadPoolExecutor(max_workers=min(_ARTICLE_FETCH_WORKERS, len(urls))) as exe:
        future_to_url = {exe.submit(_fetch_article_text, url): url for url in urls}
        for future in as_completed(future_to_url):
            body = future.result()
            if body:
                texts.append(body)

    return texts, urls


# ── Ollama extraction ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a structured data extractor for football analytics. Your task is to "
    "analyse recent news text about a national football team and extract specific signals.\n\n"
    "STRICT RULES:\n"
    "1. Only extract information EXPLICITLY stated in the provided text — never infer, "
    "assume, or draw on external knowledge.\n"
    "2. key_absences: only include players explicitly described as OUT, injured, suspended, "
    "or will miss the next match. Do NOT include players described as 'doubtful', "
    "'carrying a knock', 'a slight concern', 'returned to training', or similar.\n"
    "3. If the text contains no useful signal, return form_score=0.0, empty lists, "
    "and confidence at or below 0.15.\n"
    "4. You MUST output valid JSON matching the exact schema — no markdown fences, "
    "no explanation, no text outside the JSON object."
)

_USER_TEMPLATE = (
    "Analyse the following recent news text about {team} and extract structured form signals.\n\n"
    "CRITICAL: You are ONLY extracting data for {team}. If the article covers a match between "
    "{team} and an opponent, ignore the opponent entirely — do not list any opponent players "
    "in key_absences, do not score the opponent's morale, do not mix up the two sides.\n\n"
    "TEXT:\n{text}\n\n"
    "Return JSON with EXACTLY this schema (no extra keys):\n"
    '{{"form_score": <float -1.0 to 1.0: positive=confident/winning/key players fit, '
    "negative=poor form/morale issues/key absences, 0.0=unclear>, "
    '"key_absences": [<strings: names of {team} players confirmed absent for their NEXT match — '
    "never list players from the opposing team>, "
    '"morale_signals": [<strings: direct short phrases from text indicating {team} morale>], '
    '"tactical_notes": "<string: {team} tactical changes or coach statements, empty if none>", '
    '"confidence": <float 0.0 to 1.0: 0=team barely mentioned, 1=rich pre-match report>}}'
)


def _call_ollama(team: str, combined_text: str, model: str) -> dict:
    """POST extraction request to Ollama. Returns parsed JSON dict.

    Raises ConnectionError if Ollama is not reachable.
    Raises ValueError if response is not valid JSON.
    """
    payload = json.dumps(
        {
            "model": model,
            "stream": False,
            "think": False,  # critical: disables internal reasoning trace
            "format": "json",
            "options": {"num_predict": 400, "temperature": 0.1, "top_k": 20},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(team=team, text=combined_text),
                },
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

    return json.loads(content)


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

    return FormAnalysis(
        team=team,
        form_score=form_score,
        key_absences=absences,
        morale_signals=signals,
        tactical_notes=notes,
        confidence=confidence,
    )


def analyse_team_form(
    team: str,
    texts: list[str],
    urls: list[str] | None = None,
    model: str = DEFAULT_MODEL,
) -> FormAnalysis:
    """Extract structured form signals from article texts using Ollama.

    Returns FormAnalysis.neutral() immediately if texts is empty (no LLM call).
    Returns neutral FormAnalysis with error set on any Ollama/parse failure.
    """
    if not texts:
        return FormAnalysis.neutral(team)

    combined = "\n\n---\n\n".join(texts)[:_MAX_COMBINED_CHARS]

    try:
        raw = _call_ollama(team, combined, model)
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


# ── High-level cached entry point ─────────────────────────────────────────────


def get_team_form_analysis(
    team: str,
    force_refresh: bool = False,
    model: str = DEFAULT_MODEL,
    max_articles: int = 3,
) -> FormAnalysis:
    """Fetch news, run LLM extraction, return cached FormAnalysis.

    Cache is per-team per-calendar-day (UTC). Re-running within the same day
    returns the cached result without making any network or LLM calls.
    Pass force_refresh=True to bypass.

    Always returns a valid FormAnalysis — never raises. Degrades to
    form_score=0.0 / confidence=0.0 on any failure.
    """
    cache_path = _today_cache_path(team)

    if not force_refresh and cache_path.exists():
        try:
            with cache_path.open() as f:
                data = json.load(f)
            analysis = FormAnalysis.from_dict(data)
            print(
                f"[llm_form] Cache HIT {team!r}: "
                f"form={analysis.form_score:+.2f}, conf={analysis.confidence:.2f}",
                file=sys.stderr,
            )
            return analysis
        except Exception as exc:
            print(f"[llm_form] Cache read failed for {team!r}: {exc}", file=sys.stderr)

    print(f"[llm_form] Fetching news for {team!r}...", file=sys.stderr)
    texts, urls = fetch_team_news(team, max_articles=max_articles)

    if not texts:
        print(f"[llm_form] No articles found for {team!r} — returning neutral.", file=sys.stderr)
        analysis = FormAnalysis.neutral(team)
    else:
        print(
            f"[llm_form] {team!r}: {len(texts)} article(s) → running {model}...",
            file=sys.stderr,
        )
        analysis = analyse_team_form(team, texts, urls=urls, model=model)
        analysis.n_articles = len(texts)

    # Cache result
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(analysis.to_dict(), f, indent=2)
    except Exception as exc:
        print(f"[llm_form] WARNING: could not write cache for {team!r}: {exc}", file=sys.stderr)

    return analysis


def get_all_teams_form(
    teams: list[str],
    force_refresh: bool = False,
    model: str = DEFAULT_MODEL,
) -> dict[str, FormAnalysis]:
    """Run form analysis for a list of teams, returning {team: FormAnalysis}.

    Teams are processed sequentially (Ollama serialises GPU inference anyway).
    HTTP article fetches within each team are parallelised.
    """
    results: dict[str, FormAnalysis] = {}
    for team in teams:
        results[team] = get_team_form_analysis(team, force_refresh=force_refresh, model=model)
    return results
