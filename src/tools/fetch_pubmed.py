import xml.etree.ElementTree as ET

import requests

REQUEST_TIMEOUT = 5




def efetch_tool(id_list: list):
    """
    Fetch PubMed articles by PMIDs and return structured article data.
    """
    if not id_list:
        return {"ids": [], "articles": []}

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(str(item) for item in id_list),
        "retmode": "xml"
    }

    res = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    res.raise_for_status()

    root = ET.fromstring(res.text)
    articles = []

    for article_node in root.findall(".//PubmedArticle"):
        medline = article_node.find("MedlineCitation")
        article = medline.find("Article") if medline is not None else None

        pmid = medline.findtext("PMID", default="") if medline is not None else ""
        title_node = article.find("ArticleTitle") if article is not None else None
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        journal = article.findtext("Journal/Title", default="") if article is not None else ""

        abstract_parts = []
        if article is not None:
            for text_node in article.findall(".//AbstractText"):
                label = text_node.get("Label")
                text = "".join(text_node.itertext()).strip()
                if text:
                    abstract_parts.append(f"{label}: {text}" if label else text)

        articles.append({
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "abstract": " ".join(abstract_parts)
        })

    return {
        "ids": [str(item) for item in id_list],
        "articles": articles
    }


tool_efetch_pubmed = [
    {
        "name": "efetch",
        "description": "Fetch structured PubMed article details such as PMID, title, journal, and abstract.",
        "parameters": {
            "type": "object",
            "properties": {
                "id_list": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["id_list"]
        }
    }
]

