import json
import requests
from typing import Optional

def search_pubmed(query: str, max_results: int = 5, year: Optional[int] = None) -> str:
    """
    Tra cứu tài liệu y sinh từ PubMed (NCBI).
    """
    # Validate & Prep Term
    max_results = min(max_results, 10)
    term = f"{query.strip()} AND {year}[Date - Publication]" if year else query.strip()
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    try:
        # Lấy danh sách ID
        search_resp = requests.get(f"{base_url}esearch.fcgi", params={
            "db": "pubmed", "term": term, "retmax": max_results, "retmode": "json"
        }, timeout=10)
        search_resp.raise_for_status()
        id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return json.dumps({"message": "No results found."})

        # Lấy thông tin chi tiết
        summary_resp = requests.get(f"{base_url}esummary.fcgi", params={
            "db": "pubmed", "id": ",".join(id_list), "retmode": "json"
        }, timeout=10)
        summary_resp.raise_for_status()
        data = summary_resp.json().get("result", {})

        # Parse Results
        results = []
        for uid in id_list:
            item = data.get(uid, {})
            # Trích xuất DOI từ list articleids
            doi = next((id["value"] for id in item.get("articleids", []) if id.get("idtype") == "doi"), "")
            
            results.append({
                "uid": uid,
                "title": item.get("title", "N/A"),
                "authors": [a.get("name") for a in item.get("authors", [])],
                "pubdate": item.get("pubdate", "N/A"),
                "doi": doi
            })

        return json.dumps({"query": term, "results": results}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Request failed: {str(e)}"})
