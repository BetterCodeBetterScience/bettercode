"""Mocked version of the parametrized year-parsing test, used for the
listings manifest entry. Deterministic, offline-safe.

This is a manifest-target test only. The pedagogical version that
makes live API calls remains in test_textmining.py for readers who
want to exercise it manually.
"""

import pytest

from bettercode.testing.textmining import parse_year_from_PubMed_record


# Canned (pmid, expected_year) pairs matching the live test data in test_textmining.py.
# parse_year_from_PubMed_record reads record["pubdate"] and returns int(pubdate.split()[0]).
CASES = [
    ("17773841", 1944),
    ("13148370", 1954),
    ("14208567", 1964),
    ("4621244", 1974),
    ("6728178", 1984),
    ("10467601", 1994),
    ("15050513", 2004),
]


def _fake_record(year: int) -> dict:
    return {"pubdate": f"{year} Jan 1"}


@pytest.mark.parametrize("pmid,year", CASES)
def test_parse_year_from_pmid_parametric(pmid: str, year: int) -> None:
    record = _fake_record(year)
    assert parse_year_from_PubMed_record(record) == year
