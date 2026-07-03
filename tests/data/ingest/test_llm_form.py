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


def test_analyse_team_trajectory_empty_blocks_skips_llm_call(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_form, "_call_ollama", lambda *a, **k: calls.append(1) or {})

    result = llm_form.analyse_team_trajectory("Spain", [])

    assert calls == []
    assert result.team == "Spain"
    assert result.confidence == 0.0


def test_analyse_team_trajectory_sends_ordered_blocks_and_trajectory_prompt(monkeypatch):
    captured = {}

    def fake_call_ollama(system_prompt, user_content, model):
        captured["system_prompt"] = system_prompt
        captured["user_content"] = user_content
        return {
            "form_score": 0.3,
            "performance_context": "Improved across the group stage.",
            "key_absences": [],
            "morale_signals": [],
            "tactical_notes": "",
            "confidence": 0.6,
        }

    monkeypatch.setattr(llm_form, "_call_ollama", fake_call_ollama)

    blocks = [
        "=== Match 1 of 2 — vs Cape Verde (2026-06-13), Spain won 3-0 ===\nDominant display.",
        "=== Match 2 of 2 — vs Japan (2026-06-19), Spain drew 1-1 ===\nStruggled to break through.",
    ]
    urls = ["https://example.com/a", "https://example.com/b"]

    result = llm_form.analyse_team_trajectory("Spain", blocks, urls=urls)

    assert captured["system_prompt"] == llm_form._SYSTEM_PROMPT_TRAJECTORY
    assert "Match 1 of 2" in captured["user_content"]
    assert "Match 2 of 2" in captured["user_content"]
    assert captured["user_content"].index("Match 1 of 2") < captured["user_content"].index(
        "Match 2 of 2"
    )
    assert result.form_score == 0.3
    assert result.confidence == 0.6
    assert result.n_articles == 2
    assert result.sources == urls
