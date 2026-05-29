import requests

# define the eutils base URL globally for the module
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def get_PubMedIDs_for_query(
    query: str, retmax: int | None = None, esearch_url: str | None = None
) -> list[str]:
    """Search PubMed for a query and return the matching record IDs.

    Args:
        query: The query to search for.
        retmax: The maximum number of results to return.
        esearch_url: Override for the esearch endpoint URL.

    Returns:
        The matching PubMed IDs.
    """
    # define the base url for the pubmed search
    if esearch_url is None:
        esearch_url = f"{BASE_URL}/esearch.fcgi"

    params = format_pubmed_query_params(query, retmax=retmax)

    response = requests.get(esearch_url, params=params)

    return get_idlist_from_response(response)


def format_pubmed_query_params(query: str, retmax: int = 10000) -> dict:
    """Format a query for use with the PubMed API.

    Args:
        query: The query to format.
        retmax: The maximum number of results to return.

    Returns:
        The formatted query parameters.
    """
    # define the parameters for the search
    return {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax}


def get_idlist_from_response(response: requests.Response) -> list[str]:
    """Extract the list of PubMed IDs from an esearch response."""
    if response.status_code == 200:
        # extract the pubmed IDs from the response
        ids = response.json()["esearchresult"]["idlist"]
        return ids
    else:
        raise ValueError("Bad request")


def get_record_from_PubMedID(pmid: str, esummary_url: str | None = None) -> dict:
    """Fetch the esummary record for a single PubMed ID."""
    if esummary_url is None:
        esummary_url = f"{BASE_URL}/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"

    response = requests.get(esummary_url)

    result_json = response.json()

    if (
        response.status_code != 200
        or "result" not in result_json
        or pmid not in result_json["result"]
        or "error" in result_json["result"][pmid]
    ):
        raise ValueError("Bad request")

    return result_json["result"][pmid]


def parse_year_from_PubMed_record(pubmed_record: dict) -> int | None:
    """Return the publication year from a PubMed record, or None if absent."""
    pubdate = pubmed_record.get("pubdate")
    return int(pubdate.split()[0]) if pubdate else None
