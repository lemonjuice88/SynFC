"""
legal_documents.py — FIFA/UEFA regulation text search
==========================================================
Downloads/extracts and searches real regulation PDFs (e.g. FIFA's
Regulations on the Status and Transfer of Players) so the Legal
department can ground its answers in actual rule text instead of
general knowledge.

Setup:
    pip install pdfplumber
    Place the PDF in the project's data/ folder (or point
    LEGAL_PDF_PATH elsewhere via env var).
"""

"""
legal_documents.py — FIFA/UEFA regulation text search
==========================================================
Downloads/extracts and searches real regulation PDFs (e.g. FIFA's
Regulations on the Status and Transfer of Players) so the Legal
department can ground its answers in actual rule text instead of
general knowledge.

TWO search modes, deliberately kept side by side:
- search_regulation(): keyword/substring matching. Fast, zero setup,
  but only catches near-exact wording.
- semantic_search_regulation(): real vector search (sentence-transformers
  embeddings + FAISS). Catches DIFFERENTLY WORDED but same-MEANING
  queries (e.g. "young player money" finding a page about "training
  compensation" even though neither exact word appears) -- this is
  what actually makes it RAG rather than plain keyword lookup.

Setup:
    pip install pdfplumber sentence-transformers faiss-cpu
    Place the PDF in the project's data/ folder (or point
    LEGAL_PDF_PATH elsewhere via env var).
"""

import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Suppress sentence-transformers/huggingface_hub's own console noise
# (the "unauthenticated requests" warning, the "Loading weights" tqdm
# progress bar) -- these are the underlying libraries' own output, not
# something we print ourselves. MUST be set before importing those
# libraries below, since some of them only read these at import time.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer

try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:
    pass  # best-effort -- don't let a logging-suppression failure break the actual import

# Read once at import time -- if set, this silences the "unauthenticated
# requests" warning FOR REAL (not just hiding it -- it actually
# authenticates), and gets you higher HF Hub rate limits too. Get one
# (free, "Read" access is enough) at https://huggingface.co/settings/tokens
_HF_TOKEN = os.environ.get("HF_TOKEN") or None

# This file lives in Tools/ -- default the PDF path to the project's
# data/ folder (a sibling of Tools/), same convention as
# player_stats_dataset.py's CSV path.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)

DEFAULT_REGULATION_PATH = os.environ.get(
    "LEGAL_PDF_PATH",
    os.path.join(_PROJECT_ROOT, "data", "Regulations on the Status and Transfer of Players - January 2025 edition.pdf"),
)

# A marker string chosen specifically because it will never appear
# naturally inside the regulation text itself -- lets us tell REAL page
# breaks apart from the ordinary newlines already present within each
# page's own text when we later need to split back into pages.
_PAGE_BREAK_MARKER = "\n===PAGE_BREAK===\n"

# Small, fast, runs locally (no API call, no cost) -- good enough
# quality for this use case. ~80MB download the first time it's used,
# cached by sentence-transformers itself afterward.
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME, token=_HF_TOKEN)
    return _embedding_model


@lru_cache(maxsize=8)
def extract_regulation_text(pdf_path: str = DEFAULT_REGULATION_PATH) -> str:
    """Reads every page of the PDF at `pdf_path` and returns it as one
    string, with pages joined by _PAGE_BREAK_MARKER (not a plain "\\n" --
    see module note above). Cached: the PDF is only actually read once
    per path, no matter how many times different agents call this in
    the same run.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return _PAGE_BREAK_MARKER.join(pages_text)


@lru_cache(maxsize=8)
def _build_vector_index(pdf_path: str = DEFAULT_REGULATION_PATH):
    """Builds (once, cached) a FAISS index over every page's embedding.
    Returns (faiss_index, list_of_page_texts) so search results can be
    mapped back to their original page number/text.

    Empty pages are skipped (a blank vector carries no real meaning and
    would just be noise in the index).
    """
    full_text = extract_regulation_text(pdf_path)
    all_pages = full_text.split(_PAGE_BREAK_MARKER)

    # Keep track of each kept page's REAL page number (1-indexed,
    # matching search_regulation()'s convention) even after skipping
    # blank ones -- otherwise page numbers reported back would be wrong.
    kept_pages = [(i + 1, text.strip()) for i, text in enumerate(all_pages) if text.strip()]
    page_numbers = [p[0] for p in kept_pages]
    page_texts = [p[1] for p in kept_pages]

    model = _get_embedding_model()
    embeddings = model.encode(page_texts, show_progress_bar=False, convert_to_numpy=True)

    # Normalize so we can use FAISS's inner-product index as cosine
    # similarity (a standard trick -- cosine similarity on normalized
    # vectors IS just their dot product).
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # IP = inner product
    index.add(embeddings)

    return index, page_numbers, page_texts


def semantic_search_regulation(query: str, pdf_path: str = DEFAULT_REGULATION_PATH, top_k: int = 5) -> dict:
    """REAL vector/semantic search: embeds `query`, finds the `top_k`
    MEANING-closest pages, regardless of exact wording overlap. This is
    what actually makes this file "RAG" rather than keyword lookup --
    see module docstring.

    Example return:
        {"found": True, "results": [{"page": 9, "text": "...", "similarity": 0.71}, ...]}
    """
    if not query or not query.strip():
        return {"found": False, "reason": "Empty query."}

    index, page_numbers, page_texts = _build_vector_index(pdf_path)

    model = _get_embedding_model()
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    similarities, indices = index.search(query_vec, top_k)

    results = []
    for similarity, idx in zip(similarities[0], indices[0]):
        if idx == -1:  # FAISS returns -1 if there are fewer than top_k vectors total
            continue
        results.append({
            "page": page_numbers[idx],
            "text": page_texts[idx],
            "similarity": round(float(similarity), 3),
        })

    if not results:
        return {"found": False, "reason": "No results (index may be empty)."}

    return {"found": True, "results": results}


def search_regulation(query_terms: List[str], pdf_path: str = DEFAULT_REGULATION_PATH, max_results: int = 5) -> dict:
    """Searches the regulation document page-by-page for any of
    `query_terms` (case-insensitive substring match). Returns up to
    `max_results` matching pages.

    Example return:
        {"found": True, "total_matches": 4,
         "sample": [{"page": 9, "text": "..."}, ...]}
    """
    full_text = extract_regulation_text(pdf_path)
    pages = full_text.split(_PAGE_BREAK_MARKER)

    query_terms_lower = [t.lower() for t in query_terms if t]
    if not query_terms_lower:
        return {"found": False, "reason": "No query terms given."}

    matches = []
    for page_number, page_text in enumerate(pages, start=1):
        if any(term in page_text.lower() for term in query_terms_lower):
            matches.append({"page": page_number, "text": page_text.strip()})

    if not matches:
        return {"found": False, "reason": f"No match found for {query_terms} in regulation document."}

    return {
        "found": True,
        "total_matches": len(matches),
        "sample": matches[:max_results],
    }


if __name__ == "__main__":
    # Quick manual test:
    #   python legal_documents.py keyword training compensation
    #   python legal_documents.py semantic "money for developing young players"
    import sys
    import json

    mode = sys.argv[1] if len(sys.argv) > 1 else "keyword"
    rest = sys.argv[2:] or ["training compensation"]

    if mode == "semantic":
        result = semantic_search_regulation(" ".join(rest))
    else:
        result = search_regulation(rest)

    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])