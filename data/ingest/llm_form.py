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
3. Fetch full article body for matched items  (parallel HTTP)
4. Concatenate + truncate to 2500 chars
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

_OLLAMA_BASE = _os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_OLLAMA_TIMEOUT = 45  # seconds — allow for cold model load
_BBC_WC_RSS = "https://feeds.bbci.co.uk/sport/football/world-cup/rss.xml"
_BBC_SPORT_RSS = "https://feeds.bbci.co.uk/sport/football/rss.xml"
_GUARDIAN_WC_RSS = "https://www.theguardian.com/football/world-cup-2026/rss"
_GUARDIAN_FOOTBALL_RSS = "https://www.theguardian.com/football/rss"
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


def _clean_html(text: str) -> str:
    """Strip tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


_SUPPORTED_DOMAINS = ("bbc.", "theguardian.com")


def _fetch_article_text(url: str) -> str:
    """Fetch full body text from a BBC Sport or Guardian article. Returns '' on failure."""
    if not url or not any(d in url for d in _SUPPORTED_DOMAINS):
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
        p
        for p in cleaned
        if len(p) >= 60
        and not p.startswith("BBC")
        and "cookie" not in p.lower()
        and "subscribe" not in p.lower()
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

    Searches BBC WC RSS (primary) and BBC Sport RSS (secondary).
    Filters by case-insensitive keyword match in title or description.
    Fetches article bodies in parallel.

    Returns ([], []) if no articles found or on network failure.
    """
    search_terms = _SEARCH_TERMS.get(team, [team])
    search_lower = [s.lower() for s in search_terms]

    # Fetch all RSS feeds; WC-specific feeds listed first (more relevant)
    all_items: list[dict] = []
    for rss_url in [_BBC_WC_RSS, _GUARDIAN_WC_RSS, _BBC_SPORT_RSS, _GUARDIAN_FOOTBALL_RSS]:
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
        desc_clean = _clean_html(item["description"]).lower()
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
    # RSS descriptions are clean summaries; always available regardless of domain.
    # Guardian articles render via JS so full-page scraping returns nav boilerplate —
    # use RSS description directly for non-BBC URLs.
    rss_descs = {m["link"]: _clean_html(m["title"] + ". " + m["description"]) for m in matched}

    bbc_urls = [u for u in urls if "bbc." in u]
    other_urls = [u for u in urls if "bbc." not in u]

    texts: list[str] = []

    # BBC: fetch full article body (works well with <p> extraction)
    if bbc_urls:
        with ThreadPoolExecutor(max_workers=min(_ARTICLE_FETCH_WORKERS, len(bbc_urls))) as exe:
            future_to_url = {exe.submit(_fetch_article_text, url): url for url in bbc_urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                body = future.result()
                texts.append(body if len(body) >= 150 else rss_descs.get(url, ""))

    # Non-BBC (Guardian etc.): use RSS description directly — it's a clean summary
    for url in other_urls:
        desc = rss_descs.get(url, "")
        if desc:
            texts.append(desc)

    texts = [t for t in texts if t]
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
    "2. performance_context: ALWAYS fill this. Summarise what the articles say about this "
    "team's form and momentum. If genuinely nothing is said, explain what the articles "
    "covered instead.\n"
    "3. form_score [-1, 1]: overall momentum signal. Positive = playing well / rising / "
    "confident. Negative = struggling / unsettled / weakened. 0 = nothing to go on.\n"
    "4. confidence [0, 1]: how much evidence you have. 0 = team barely mentioned. "
    "0.5 = some signal. 0.9+ = multiple clear statements about form and momentum.\n"
    "5. Output valid JSON only — no markdown, no text outside the JSON object."
)

_USER_TEMPLATE = (
    "Analyse the following recent news text about {team} and extract structured form signals.\n\n"
    "CRITICAL: You are ONLY extracting data for {team}. If the article covers a match between "
    "{team} and an opponent, ignore the opponent entirely — do not list any opponent players "
    "in key_absences, do not score the opponent's morale, do not mix up the two sides.\n\n"
    "TEXT:\n{text}\n\n"
    "Return JSON with EXACTLY this schema (no extra keys):\n"
    '{{"form_score": <float -1.0 to 1.0: overall narrative sentiment for {team} — '
    "factor in performance quality, momentum, morale, and absences together>, "
    '"performance_context": "<string: REQUIRED — 1-3 sentences on {team}\'s current form '
    "narrative (performance quality, momentum, morale, press perception) OR, if no signal, "
    'a brief explanation of what the articles covered and why no form signal could be extracted>", '
    '"key_absences": [<strings: names of {team} players confirmed absent for their NEXT match — '
    "never list players from the opposing team>], "
    '"morale_signals": [<strings: direct short phrases from text indicating {team} morale or '
    "team atmosphere>], "
    '"tactical_notes": "<string: {team} tactical changes or coach statements, empty if none>", '
    '"confidence": <float 0.0 to 1.0: how strongly the text supports your form_score. '
    "This value scales how much your judgement moves the prediction, so be calibrated: "
    "0=team barely mentioned or no clear performance signal; 0.3-0.5=some signal but mixed or "
    "thin; 0.7-1.0=multiple clear, decisive statements about how the team is playing. "
    "Reserve high confidence for genuinely strong, well-sourced evidence>}}"
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
) -> dict[str, FormAnalysis]:
    """Fetch articles for both teams combined, extract signals for each in one LLM call.

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
    home_fa = analyse_team_form(home, home_texts, urls=home_urls, model=model)
    away_fa = analyse_team_form(away, away_texts, urls=away_urls, model=model)
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
    match_pairs: list[tuple[str, str]],
    model: str = DEFAULT_MODEL,
) -> dict[str, FormAnalysis]:
    """Run combined match-level form analysis for a list of (home, away) pairs.

    Each pair is one LLM call reading articles for both teams simultaneously.
    Returns {team_name: FormAnalysis} for all teams across all pairs.
    """
    results: dict[str, FormAnalysis] = {}
    for home, away in match_pairs:
        pair_results = get_match_form_analysis(home, away, model=model)
        results.update(pair_results)
    return results
