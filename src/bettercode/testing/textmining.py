import requests

# define the eutils base URL globally for the module
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def get_PubMedIDs_for_query(
    query: str, retmax: int = None, esearch_url: str = None
) -> list:
    """
    Search database for a given query and return a list of IDs.
    :param query: str, the query to search for
    :param retmax: int, the maximum number of results to return
    :base_url: str, the base url for the pubmed search
    :return: list, a list of pubmed IDs
    """
    # define the base url for the pubmed search
    if esearch_url is None:
        esearch_url = f"{BASE_URL}/esearch.fcgi"

    params = format_pubmed_query_params(query, retmax=retmax)

    response = requests.get(esearch_url, params=params)

    return get_idlist_from_response(response)


def format_pubmed_query_params(query: str, retmax: int = 10000) -> str:
    """
    Format a query for use with the pubmed api.
    :param query: str, the query to format
    :return: dict, the formatted query dict
    """

    # define the parameters for the search
    return {"db": "pubmed", "term": query, "retmode": "json", "retmax": retmax}


def get_idlist_from_response(response: requests.Response) -> list:
    if response.status_code == 200:
        # extract the pubmed IDs from the response
        ids = response.json()["esearchresult"]["idlist"]
        return ids
    else:
        raise ValueError("Bad request")


def get_record_from_PubMedID(pmid: str, esummary_url: str = None) -> dict:

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


def parse_year_from_PubMed_record(pubmed_record: dict) -> int:
    pubdate = pubmed_record.get("pubdate")
    return int(pubdate.split()[0]) if pubdate else None
