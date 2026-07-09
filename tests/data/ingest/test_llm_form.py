from __future__ import annotations

import json

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


def test_strip_markdown_fence_removes_json_fence():
    fenced = '```json\n{"form_score": 0.5, "confidence": 0.8}\n```'
    assert llm_form._strip_markdown_fence(fenced) == '{"form_score": 0.5, "confidence": 0.8}'


def test_strip_markdown_fence_removes_plain_fence():
    fenced = '```\n{"form_score": 0.5}\n```'
    assert llm_form._strip_markdown_fence(fenced) == '{"form_score": 0.5}'


def test_strip_markdown_fence_leaves_unfenced_content_unchanged():
    raw = '{"form_score": 0.5}'
    assert llm_form._strip_markdown_fence(raw) == raw


def test_call_ollama_parses_fenced_json_response(monkeypatch):
    """qwen3.5:9b has been observed wrapping JSON in a markdown fence despite
    format: "json" — _call_ollama must still parse it rather than raising."""

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fenced_content = '```json\n{"form_score": 0.9, "confidence": 0.95}\n```'
    ollama_response = json.dumps(
        {"message": {"content": fenced_content}, "done_reason": "stop"}
    ).encode()

    monkeypatch.setattr(
        llm_form.urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(ollama_response),
    )

    raw = llm_form._call_ollama("system", "user", "qwen3.5:9b")
    assert raw == {"form_score": 0.9, "confidence": 0.95}


def test_analyse_team_trajectory_empty_blocks_skips_llm_call(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_form, "_call_ollama", lambda *a, **k: calls.append(1) or {})

    result = llm_form.analyse_team_trajectory("Spain", [])

    assert calls == []
    assert result.team == "Spain"
    assert result.confidence == 0.0


def test_analyse_team_trajectory_sends_ordered_blocks_and_trajectory_prompt(monkeypatch):
    captured = {}

    def fake_call_ollama(system_prompt, user_content, model, num_predict=400, num_ctx=8192):
        captured["system_prompt"] = system_prompt
        captured["user_content"] = user_content
        captured["num_predict"] = num_predict
        captured["num_ctx"] = num_ctx
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
    # Larger output budget than the single-match path's default (400) — the
    # trajectory prompt asks for a longer, multi-match synthesis.
    assert captured["num_predict"] == llm_form._TRAJECTORY_NUM_PREDICT
    assert captured["num_predict"] > 400
    # Large enough context window to fit every match's full article text —
    # see _TRAJECTORY_NUM_CTX for the empirical GPU-memory reasoning.
    assert captured["num_ctx"] == llm_form._TRAJECTORY_NUM_CTX


def test_analyse_team_trajectory_does_not_truncate_later_matches(monkeypatch):
    """Regression test for the real bug found in production: a naive
    join-then-slice on the combined text silently dropped every match after
    the first two once their articles alone exceeded the old character cap
    — a 5-match trajectory (Morocco) never saw its final 3-0 win over Canada
    because that match's content was truncated away before reaching the LLM.
    """
    captured = {}

    def fake_call_ollama(system_prompt, user_content, model, num_predict=400, num_ctx=8192):
        captured["user_content"] = user_content
        return {
            "form_score": 0.5,
            "performance_context": "ok",
            "key_absences": [],
            "morale_signals": [],
            "tactical_notes": "",
            "confidence": 0.7,
        }

    monkeypatch.setattr(llm_form, "_call_ollama", fake_call_ollama)

    # Two large early matches whose combined size alone would have exceeded
    # the old 16000-char cap, plus a distinctly-labeled later match.
    huge_article = "Dense match report text. " * 1000  # ~26,000 chars
    blocks = [
        f"=== Match 1 of 3 — vs Brazil (2026-06-13), Morocco won 2-1 ===\n{huge_article}",
        f"=== Match 2 of 3 — vs Scotland (2026-06-19), Morocco drew 0-0 ===\n{huge_article}",
        "=== Match 3 of 3 — vs Canada (2026-07-04), Morocco won 3-0 ===\nDominant win over Canada.",
    ]

    llm_form.analyse_team_trajectory("Morocco", blocks)

    assert "Match 1 of 3" in captured["user_content"]
    assert "Match 2 of 3" in captured["user_content"]
    assert "Match 3 of 3" in captured["user_content"]
    assert "Dominant win over Canada" in captured["user_content"]
