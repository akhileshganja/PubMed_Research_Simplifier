"""PubMed API client for data ingestion - esearch + efetch."""

import requests
import xmltodict
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential

from pubmed_nlp.config import get_settings


@dataclass
class PubMedArticle:
    """Structured PubMed article data."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    year: Optional[int]
    mesh_terms: List[str]
    keywords: List[str]
    journal: str
    doi: Optional[str]
    publication_types: List[str]


class PubMedClient:
    """NCBI E-utilities client for PubMed data retrieval."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self):
        settings = get_settings()
        self.email = settings.pubmed_email
        self.api_key = settings.pubmed_api_key
        self.max_results = settings.max_articles_per_query
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": f"PubMedNLP/1.0 ({self.email})"})
    
    def _get_params(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build request parameters with API key if available."""
        params = {"email": self.email, "retmode": "xml"}
        if self.api_key:
            params["api_key"] = self.api_key
        if extra:
            params.update(extra)
        return params
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def esearch(self, query: str, max_results: int = None) -> List[str]:
        """Search PubMed and return article IDs."""
        max_results = max_results or self.max_results
        
        params = self._get_params({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": "relevance",
            "usehistory": "y"
        })
        
        response = self._session.get(f"{self.BASE_URL}/esearch.fcgi", params=params, timeout=30)
        response.raise_for_status()
        
        data = xmltodict.parse(response.content)
        idlist = data.get("eSearchResult", {}).get("IdList", {}).get("Id", [])
        
        return idlist if isinstance(idlist, list) else [idlist] if idlist else []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def efetch(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch article details by PubMed IDs."""
        if not pmids:
            return []
        
        params = self._get_params({
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract"
        })
        
        response = self._session.get(f"{self.BASE_URL}/efetch.fcgi", params=params, timeout=60)
        response.raise_for_status()
        
        data = xmltodict.parse(response.content)
        articles = data.get("PubmedArticleSet", {}).get("PubmedArticle", [])
        
        if not isinstance(articles, list):
            articles = [articles]
        
        return [self._parse_article(a) for a in articles if a]
    
    def _parse_article(self, article_data: Dict) -> PubMedArticle:
        """Parse XML article data into structured format."""
        medline = article_data.get("MedlineCitation", {})
        article = medline.get("Article", {})
        
        pmid = str(medline.get("PMID", {}).get("#text", ""))
        title = article.get("ArticleTitle", "")
        
        # Abstract handling
        abstract_data = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_data, list):
            abstract = " ".join(str(a.get("#text", a)) if isinstance(a, dict) else str(a) for a in abstract_data)
        elif isinstance(abstract_data, dict):
            abstract = abstract_data.get("#text", "")
        else:
            abstract = str(abstract_data)
        
        # Authors
        authors = []
        author_list = article.get("AuthorList", {}).get("Author", [])
        if not isinstance(author_list, list):
            author_list = [author_list]
        for author in author_list:
            if isinstance(author, dict):
                last = author.get("LastName", "")
                first = author.get("ForeName", "")
                if last:
                    authors.append(f"{first} {last}".strip())
        
        # Year
        year = None
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year_str = pub_date.get("Year") or pub_date.get("MedlineDate", "")[:4]
        if year_str and year_str.isdigit():
            year = int(year_str)
        
        # MeSH Terms
        mesh_terms = []
        mesh_list = medline.get("MeshHeadingList", {}).get("MeshHeading", [])
        if not isinstance(mesh_list, list):
            mesh_list = [mesh_list]
        for mesh in mesh_list:
            if isinstance(mesh, dict):
                desc = mesh.get("DescriptorName", {})
                term = desc.get("#text", "") if isinstance(desc, dict) else str(desc)
                if term:
                    mesh_terms.append(term)
        
        # Keywords
        keywords = []
        keyword_list = medline.get("KeywordList", {}).get("Keyword", [])
        if not isinstance(keyword_list, list):
            keyword_list = [keyword_list]
        for kw in keyword_list:
            if isinstance(kw, dict):
                keywords.append(kw.get("#text", ""))
            else:
                keywords.append(str(kw))
        
        journal = article.get("Journal", {}).get("Title", "")
        
        # DOI
        doi = None
        ids = article.get("ELocationID", [])
        if not isinstance(ids, list):
            ids = [ids]
        for eid in ids:
            if isinstance(eid, dict) and eid.get("@EIdType") == "doi":
                doi = eid.get("#text")
                break
        
        # Publication Types
        pub_types = []
        pt_list = article.get("PublicationTypeList", {}).get("PublicationType", [])
        if not isinstance(pt_list, list):
            pt_list = [pt_list]
        for pt in pt_list:
            if isinstance(pt, dict):
                pub_types.append(pt.get("#text", ""))
            else:
                pub_types.append(str(pt))
        
        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            mesh_terms=mesh_terms,
            keywords=keywords,
            journal=journal,
            doi=doi,
            publication_types=pub_types
        )
    
    def search_and_fetch(self, query: str, max_results: int = None) -> List[PubMedArticle]:
        """Complete pipeline: search then fetch articles."""
        pmids = self.esearch(query, max_results)
        if not pmids:
            return []
        return self.efetch(pmids)
