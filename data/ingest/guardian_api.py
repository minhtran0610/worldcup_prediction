"""Guardian Content API client — full-text article search with date filtering.

Used to retroactively recover post-match analysis for WC2026 matches whose
reports have already rotated out of the RSS feeds llm_form.py otherwise
relies on (RSS exposes only currently-live items, no archive/search).

Requires a free Guardian Content API developer key:
https://bonobo.capi.gutools.co.uk/register/developer
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import requests

_BASE_URL: str = "https://content.guardianapis.com/search"
_TIMEOUT_SECONDS: int = 20


def get_api_key() -> str | None:
    """Return GUARDIAN_API_KEY from environment, or None if not set."""
    return os.environ.get("GUARDIAN_API_KEY")


def search_articles(
    query: str,
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
    api_key: str | None = None,
    page_size: int = 5,
) -> list[dict]:
    """Full-text search the Guardian Content API within a date range.

    Returns a list of {title, url, published, body_html} dicts (body_html is
    the raw article body HTML — cleaning is the caller's responsibility, so
    HTML-stripping logic stays in one place). Returns [] on any failure
    (missing key, network error, empty results) so callers can degrade
    gracefully.
    """
    if api_key is None:
        api_key = get_api_key()
    if not api_key:
        print("[guardian_api] GUARDIAN_API_KEY not set — skipping search.", file=sys.stderr)
        return []

    params = {
        "q": query,
        "from-date": from_date.strftime("%Y-%m-%d"),
        "to-date": to_date.strftime("%Y-%m-%d"),
        "show-fields": "body",
        "page-size": page_size,
        "api-key": api_key,
    }

    try:
        resp = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[guardian_api] Request failed: {exc}", file=sys.stderr)
        return []

    try:
        data: dict = resp.json()
    except ValueError as exc:
        print(f"[guardian_api] JSON parse error: {exc}", file=sys.stderr)
        return []

    results = data.get("response", {}).get("results", [])
    articles: list[dict] = []
    for r in results:
        body_html = r.get("fields", {}).get("body", "")
        articles.append(
            {
                "title": r.get("webTitle", ""),
                "url": r.get("webUrl", ""),
                "published": r.get("webPublicationDate", ""),
                "body_html": body_html,
            }
        )
    return articles
