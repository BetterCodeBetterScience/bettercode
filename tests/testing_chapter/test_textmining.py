import pytest
import requests
import time

from bettercode.testing.textmining import (
    get_PubMedIDs_for_query,
    get_record_from_PubMedID,
    parse_year_from_PubMed_record,
)

# --- Fixtures ---


@pytest.fixture(scope="session")
def ids():
    query = "friston-k AND 'free energy'"
    ids = get_PubMedIDs_for_query(query)
    return ids


@pytest.fixture(scope="session")
def valid_pmid():
    return "39312494"


@pytest.fixture(scope="session")
def pmid_record(valid_pmid):
    record = get_record_from_PubMedID(valid_pmid)
    return record


# --- Tests for get_PubMedIDs_for_query ---


def test_get_PubMedIDs_for_query_check_valid(ids):
    query = "friston-k AND 'free energy'"
    ids = get_PubMedIDs_for_query(query)

    # make sure that a list is returned
    assert isinstance(ids, list)
    # make sure the list is not empty
    assert len(ids) > 0


def test_get_PubMedIDs_for_query_check_empty():
    query = "friston-k AND 'fizzbuzz'"
    ids = get_PubMedIDs_for_query(query)

    # make sure that a list is returned
    assert isinstance(ids, list)
    # make sure the resulting list is empty
    assert len(ids) == 0


def test_get_PubMedIDs_for_query_check_badurl():
    query = "friston-k AND 'free energy'"
    # bad url
    esearch_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.f'

    # make sure that the function raises an exception
    with pytest.raises(Exception):
        ids = get_PubMedIDs_for_query(query, esearch_url=esearch_url)


# --- Mock fixture (improved version) ---


@pytest.fixture
def mock_pubmed_api(monkeypatch):

    class MockPubMedResponse:
        status_code = 200

        def json(self):
            return {
                'header': {'type': 'esearch', 'version': '0.3'},
                'esearchresult': {
                    'count': '2',
                    'retmax': '20',
                    'retstart': '0',
                    'idlist': ['39312494', '39089179']
                }
            }

    def mock_get(*args, **kwargs):
        return MockPubMedResponse()

    # Apply the monkeypatch for requests.get to mock_get
    monkeypatch.setattr(requests, "get", mock_get)


# The test requests the setup, then performs the action and assertion.
def test_get_PubMedIDs_for_query_check_valid_mocked(mock_pubmed_api):
    # Action: Call the function under test
    query = "friston-k AND 'free energy'"
    ids = get_PubMedIDs_for_query(query)

    # Assertion: Check the result
    assert isinstance(ids, list)
    assert len(ids) == 2


# --- Tests for get_record_from_PubMedID ---


def test_get_record_from_valid_PubMedID(pmid_record, valid_pmid):
    assert pmid_record is not None
    assert isinstance(pmid_record, dict)
    assert pmid_record['uid'] == valid_pmid


def test_get_record_from_invalid_PubMedID():
    pmid = "10000000000"
    with pytest.raises(ValueError):
        record = get_record_from_PubMedID(pmid)


# --- Tests for parse_year_from_PubMed_record ---


def test_parse_year_from_PubMed_record():
    record = {
        "pubdate": "2021 Jan 1"
    }
    year = parse_year_from_PubMed_record(record)
    assert year == 2021


def test_parse_year_from_PubMed_record_empty():
    record = {
        "pubdate": ""
    }
    year = parse_year_from_PubMed_record(record)
    assert year is None


# --- Parametrized tests ---


testdata = [
    ('17773841', 1944),
    ('13148370', 1954),
    ('14208567', 1964),
    ('4621244', 1974),
    ('6728178', 1984),
    ('10467601', 1994),
    ('15050513', 2004)
]


@pytest.mark.parametrize("pmid, year_true", testdata)
def test_parse_year_from_pmid_parametric(pmid, year_true):
    time.sleep(0.5)  # delay to avoid hitting the PubMed API too quickly
    record = get_record_from_PubMedID(pmid)
    year_result = parse_year_from_PubMed_record(record)
    assert year_result == year_true
