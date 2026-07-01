from __future__ import annotations

import pandas as pd

import data.ingest.llm_form as llm_form


def test_fetch_team_match_report_filters_to_team_title_matches(monkeypatch):
    def fake_search(query, from_date, to_date, api_key=None):
        assert query == "Spain Croatia"
        return [
            {
                "title": "Spain thrash Croatia in group stage rout",
                "url": "https://theguardian.com/spain-croatia",
                "published": "2026-06-15T22:00:00Z",
                "body_html": "<p>Spain were utterly dominant throughout.</p>",
            },
            {
                "title": "Transfer news: unrelated gossip column",
                "url": "https://theguardian.com/unrelated",
                "published": "2026-06-15T10:00:00Z",
                "body_html": "<p>Nothing to do with this match.</p>",
            },
        ]

    monkeypatch.setattr(llm_form.guardian_api, "search_articles", fake_search)

    texts, urls = llm_form.fetch_team_match_report("Spain", "Croatia", pd.Timestamp("2026-06-15"))
    assert len(texts) == 1
    assert "dominant" in texts[0]
    assert urls == ["https://theguardian.com/spain-croatia"]


def test_fetch_team_match_report_returns_empty_when_no_results(monkeypatch):
    def fake_search(query, from_date, to_date, api_key=None):
        return []

    monkeypatch.setattr(llm_form.guardian_api, "search_articles", fake_search)

    texts, urls = llm_form.fetch_team_match_report("Spain", "Croatia", pd.Timestamp("2026-06-15"))
    assert texts == []
    assert urls == []


def test_clean_html_strips_tags_and_entities():
    assert llm_form.clean_html("<p>Spain &amp; Croatia</p>") == "Spain & Croatia"
