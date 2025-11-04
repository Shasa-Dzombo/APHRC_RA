import requests
from bs4 import BeautifulSoup 
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any, Optional, Union
import time
import json
import os
from dotenv import load_dotenv
import re
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# -----------------------
# Global model cache to avoid reloading
# -----------------------
_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is None:
        print("Loading SentenceTransformer model (one-time setup)...")
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache

# -----------------------
# Fetch from CrossRef
# -----------------------
def fetch_crossref(query: str, rows: int = 5) -> List[Dict[str, Any]]:
    """Fetch papers from CrossRef API"""
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": rows}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return []

        data = response.json()
        results = []
        for item in data.get("message", {}).get("items", []):
            results.append({
                "source": "CrossRef",
                "title": item.get("title", [""])[0],
                "abstract": item.get("abstract", ""),
                "authors": [f"{a.get('given', '')} {a.get('family', '')}" for a in item.get("author", [])] if "author" in item else [],
                "year": item.get("issued", {}).get("date-parts", [[None]])[0][0],
                "citations": item.get("is-referenced-by-count", 0),
                "venue": item.get("container-title", [""])[0],
                "url": item.get("URL", "")
            })
        return results
    except Exception as e:
        print(f"CrossRef API error: {str(e)}")
        return []

# -----------------------
# Fetch from Semantic Scholar
# -----------------------
def fetch_semantic_scholar(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch papers from Semantic Scholar API"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,citationCount,url,venue"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return []

        data = response.json()
        results = []
        for paper in data.get("data", []):
            results.append({
                "source": "Semantic Scholar",
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "year": paper.get("year"),
                "citations": paper.get("citationCount", 0),
                "venue": paper.get("venue"),
                "url": paper.get("url", "")
            })
        return results
    except Exception as e:
        print(f"Semantic Scholar API error: {str(e)}")
        return []

# -----------------------
# Google Scholar API
# -----------------------

def fetch_google_scholar(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch papers from Google Scholar API (via Serper.dev)"""

    load_dotenv()
    api_key = os.getenv("GOOGLE_SK", "")
    url = "https://google.serper.dev/scholar"

    payload = json.dumps({
    "q": query,
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        # Use the POST request result that was already made
        if response.status_code == 200:
            data = response.json()
            results = []
            for paper in data.get("organic", []):  # Changed from "data" to "organic" based on actual response structure
                results.append({
                    "source": "Google Scholar",
                    "title": paper.get("title", ""),
                    "snippet": paper.get("snippet", ""),  # Changed from "abstract" to "snippet" based on API response
                    "link": paper.get("link", ""),  # Changed from "url" to "link" based on API response
                    "year": paper.get("year", ""),
                    "citations": paper.get("cited_by", {}).get("total", 0) if "cited_by" in paper else 0
                })
            print(f"Found {len(results)} papers")
            return results
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return []
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        return []


# -----------------------
# Merge and rank results
# -----------------------
def merge_results(*sources) -> List[Dict[str, Any]]:
    """Merge results from multiple sources"""
    combined = []
    for src in sources:
        combined.extend(src)
    return combined

def rank_by_relevance(query: str, papers: List[Dict[str, Any]], model=None) -> List[Dict[str, Any]]:
    """Rank papers by relevance with hierarchical confidence scoring"""
    if not papers:
        return []
        
    if model is None:
        model = load_model()
        
    try:
        query_emb = model.encode(query, convert_to_tensor=True)
        
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            text = f"{title} {abstract}"
            
            # Avoid encoding empty texts
            if not text.strip():
                paper["relevance"] = 0
                paper["confidence_tier"] = "low"
                continue
                
            paper_emb = model.encode(text, convert_to_tensor=True)
            semantic_similarity = util.cos_sim(query_emb, paper_emb).item()
            
            # Enhanced weighted scoring with stronger hierarchy
            recency_factor = 1.0
            if paper["year"]:
                try:
                    years_old = max(1, 2025 - int(paper["year"]))
                    # More aggressive recency weighting for better hierarchy
                    recency_factor = min(2.0, 1 + (5 / years_old))
                except (ValueError, TypeError):
                    pass
            
            # Enhanced citation factor with logarithmic scaling
            citation_count = paper.get("citations", 0) or 0
            citation_factor = 1 + (0.1 * (citation_count ** 0.3)) if citation_count > 0 else 1
            
            # Calculate final relevance score
            final_score = semantic_similarity * recency_factor * citation_factor
            paper["relevance"] = final_score
            
            # Assign confidence tiers based on score thresholds for clear hierarchy
            if final_score >= 0.8:
                paper["confidence_tier"] = "highest"
                paper["hierarchy_rank"] = 1
            elif final_score >= 0.6:
                paper["confidence_tier"] = "high"
                paper["hierarchy_rank"] = 2
            elif final_score >= 0.4:
                paper["confidence_tier"] = "medium"
                paper["hierarchy_rank"] = 3
            elif final_score >= 0.2:
                paper["confidence_tier"] = "low"
                paper["hierarchy_rank"] = 4
            else:
                paper["confidence_tier"] = "very_low"
                paper["hierarchy_rank"] = 5
        
        # Sort by relevance score (highest first) for clear hierarchy
        papers.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        # Add position indicators for the hierarchy
        for i, paper in enumerate(papers):
            paper["position"] = i + 1
            if i == 0:
                paper["is_primary"] = True
                paper["tier_label"] = "Primary Reference (Highest Confidence)"
            elif i < 3:
                paper["is_primary"] = False
                paper["tier_label"] = f"Secondary Reference #{i} (High Confidence)"
            else:
                paper["is_primary"] = False
                paper["tier_label"] = f"Supporting Reference #{i} (Supporting Evidence)"
        
        # Log the hierarchy for debugging
        if papers:
            print(f"Literature Hierarchy - Primary: {papers[0].get('title', 'Unknown')[:50]}... (Score: {papers[0].get('relevance', 0):.3f})")
            if len(papers) > 1:
                print(f"Secondary references: {len(papers)-1} papers with scores {papers[1].get('relevance', 0):.3f} to {papers[-1].get('relevance', 0):.3f}")
        
        return papers
    except Exception as e:
        print(f"Error ranking papers: {str(e)}")
        return papers

def search_literature(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search and rank papers from multiple academic sources with relevance scoring"""
    
    print(f"Searching literature for: '{query[:50]}...' (limit: {limit})")
    start_time = time.time()
    
    # Fetch from both academic sources for comprehensive results
    ss_results = fetch_semantic_scholar(query, limit=limit)
    cr_results = fetch_crossref(query, rows=limit)
    
    # Combine results from both sources
    combined = merge_results(ss_results, cr_results)
    print(f"Got {len(ss_results)} from Semantic Scholar + {len(cr_results)} from CrossRef")
    
    if not combined:
        print("No literature found for this query")
        return []
    
    # Load model and rank by relevance for best quality results
    model = load_model()
    ranked = rank_by_relevance(query, combined, model)
    
    # Return top results
    result = ranked[:limit]
    
    end_time = time.time()
    print(f"Literature search completed in {end_time - start_time:.2f} seconds")
    print(f"Returning {len(result)} relevant papers")
    
    return result

def search_literature_with_ranking(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Slower but higher quality search with relevance ranking"""
    model = load_model()
    
    ss_results = fetch_semantic_scholar(query, limit=limit)
    cr_results = fetch_crossref(query, rows=limit)
    
    combined = merge_results(ss_results, cr_results)
    ranked = rank_by_relevance(query, combined, model)
    
    return ranked[:limit]

async def fetch_webpage(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search and scrape web content related to a research query.
    
    Args:
        query: The research question or topic to search for
        max_results: Maximum number of web pages to scrape (default: 3)
    
    Returns:
        List of dictionaries containing scraped information
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_SK", "")
    
    try:
        # First, get relevant URLs using Google Search API
        search_url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "num": max_results
        })
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        logging.info(f"Searching web content for: {query}")
        response = requests.request("POST", search_url, headers=headers, data=payload)
        
        if response.status_code != 200:
            logging.error(f"Search API error: {response.status_code}")
            return []
            
        search_results = response.json()
        web_data = []
        
        for result in search_results.get("organic", [])[:max_results]:
            url = result.get("link")
            if not url or not is_safe_url(url):
                continue
                
            try:
                # Fetch webpage content
                page_response = requests.get(
                    url, 
                    headers={'User-Agent': 'Research Bot 1.0'},
                    timeout=10
                )
                if page_response.status_code != 200:
                    continue
                    
                # Parse content
                soup = BeautifulSoup(page_response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                # Extract main content
                content = extract_main_content(soup)
                
                # Structure the data
                data = {
                    "url": url,
                    "title": get_clean_text(soup.title.string) if soup.title else "",
                    "source": urlparse(url).netloc,
                    "content": content,
                    "date_accessed": time.strftime("%Y-%m-%d"),
                    "snippet": result.get("snippet", ""),
                    "relevance_score": calculate_relevance_score(content, query)
                }
                
                if data["content"]:  # Only add if we got meaningful content
                    web_data.append(data)
                    logging.info(f"Successfully scraped content from {url}")
                    
            except Exception as e:
                logging.warning(f"Error scraping {url}: {str(e)}")
                continue
                
        return web_data
        
    except Exception as e:
        logging.error(f"Error in fetch_webpage: {str(e)}")
        return []

def is_safe_url(url: str) -> bool:
    """Check if URL is safe to scrape"""
    parsed = urlparse(url)
    return all([
        parsed.scheme in ['http', 'https'],
        not any(bad in parsed.netloc for bad in ['localhost', '127.0.0.1', '.local']),
        any(tld in parsed.netloc for tld in ['.org', '.edu', '.gov', '.com', '.net'])
    ])

def get_clean_text(text: Optional[str]) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main content from webpage while filtering boilerplate"""
    # Try to find main content containers
    main_tags = soup.find_all(['article', 'main', 'div'], class_=re.compile(r'content|article|main|post'))
    
    if not main_tags:
        # Fallback to paragraph extraction
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 100)
    else:
        content = ' '.join(tag.get_text().strip() for tag in main_tags)
    
    # Clean the content
    content = get_clean_text(content)
    
    # Truncate if too long (prevent token limit issues)
    max_chars = 5000
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
        
    return content

def calculate_relevance_score(content: str, query: str) -> float:
    """Calculate relevance score of content to query"""
    try:
        # Use the existing sentence transformer model
        model = load_model()
        query_emb = model.encode(query, convert_to_tensor=True)
        content_emb = model.encode(content[:1000], convert_to_tensor=True)  # Use first 1000 chars for efficiency
        
        return float(util.cos_sim(query_emb, content_emb).item())
    except Exception:
        return 0.0
