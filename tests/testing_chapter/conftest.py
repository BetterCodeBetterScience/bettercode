"""Shared pytest fixtures for tests/testing_chapter/.

The _mock_pubmed_network autouse fixture replaces requests.get with canned
PubMed responses, making the entire suite deterministic and offline-safe.
"""

import re
from unittest.mock import patch

import pytest
import requests


# Map known PMIDs (from the parametric test + valid_pmid fixture) to canned years.
_PMID_YEAR = {
    "17773841": "1944",
    "13148370": "1954",
    "14208567": "1964",
    "4621244": "1974",
    "6728178": "1984",
    "10467601": "1994",
    "15050513": "2004",
    "39312494": "2024",  # the valid_pmid fixture
    "39089179": "2024",  # second ID returned by the canned esearch response
}


class _MockResponse:
    def __init__(self, status_code: int, data: dict) -> None:
        self.status_code = status_code
        self._data = data

    def json(self) -> dict:
        return self._data


def _mock_get(url: str, *args, **kwargs) -> _MockResponse:
    """Return canned PubMed API responses based on URL patterns."""
    # esearch endpoint
    if "esearch.fcgi" in url:
        # A query containing 'fizzbuzz' should produce empty results
        params = kwargs.get("params", {})
        term = params.get("term", "") if isinstance(params, dict) else ""
        if "fizzbuzz" in term.lower():
            return _MockResponse(200, {"esearchresult": {"idlist": []}})
        return _MockResponse(
            200,
            {"esearchresult": {"idlist": ["39312494", "39089179"]}},
        )

    # esummary endpoint
    if "esummary.fcgi" in url:
        m = re.search(r"id=(\d+)", url)
        pmid = m.group(1) if m else "0"
        if pmid == "10000000000":
            # The "invalid PMID" test path: result with error
            return _MockResponse(
                200, {"result": {pmid: {"error": "cannot get document"}}}
            )
        year = _PMID_YEAR.get(pmid, "2020")
        return _MockResponse(
            200,
            {"result": {pmid: {"uid": pmid, "pubdate": f"{year} Jan 1"}}},
        )

    # Any other URL (e.g. the truncated bad-url test) returns 404
    return _MockResponse(404, {})


@pytest.fixture(scope="session", autouse=True)
def _mock_pubmed_network_session():
    """Session-scoped autouse fixture: patch requests.get for the entire test
    session so that session-scoped fixtures (ids, pmid_record) also use the
    canned responses.

    Uses unittest.mock.patch directly because monkeypatch is function-scoped.
    """
    with patch.object(requests, "get", side_effect=_mock_get):
        yield
