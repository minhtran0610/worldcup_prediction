from __future__ import annotations

import pandas as pd

import data.ingest.guardian_api as guardian_api


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_data


def test_search_articles_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)
    result = guardian_api.search_articles(
        "Spain Croatia", pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-17")
    )
    assert result == []


def test_search_articles_parses_response(monkeypatch):
    monkeypatch.setenv("GUARDIAN_API_KEY", "fake-key")

    def fake_get(url, params=None, timeout=None):
        assert params["q"] == "Spain Croatia"
        assert params["from-date"] == "2026-06-15"
        assert params["to-date"] == "2026-06-17"
        return _FakeResponse(
            {
                "response": {
                    "results": [
                        {
                            "webTitle": "Spain thrash Croatia",
                            "webUrl": "https://theguardian.com/x",
                            "webPublicationDate": "2026-06-15T22:00:00Z",
                            "fields": {"body": "<p>Spain were dominant.</p>"},
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(guardian_api.requests, "get", fake_get)

    result = guardian_api.search_articles(
        "Spain Croatia", pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-17")
    )
    assert len(result) == 1
    assert result[0]["title"] == "Spain thrash Croatia"
    assert result[0]["body_html"] == "<p>Spain were dominant.</p>"


def test_search_articles_returns_empty_on_request_failure(monkeypatch):
    monkeypatch.setenv("GUARDIAN_API_KEY", "fake-key")

    def fake_get(url, params=None, timeout=None):
        raise guardian_api.requests.exceptions.RequestException("boom")

    monkeypatch.setattr(guardian_api.requests, "get", fake_get)

    result = guardian_api.search_articles(
        "Spain Croatia", pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-17")
    )
    assert result == []
