"""
buywise-flask-rag/app.py

Optimized Flask RAG app with:
- scikit-learn NearestNeighbors (Windows-friendly)
- OpenAI v1 client
- Defensive error handling
- Supabase Google login (optional)
- CORS enabled for Vercel + localhost
- Parallel SerpAPI + scraping
- Embedding cache for speed
- Standardized JSON responses
- Endpoints: /search, /recommendations
"""

import os
import re
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from openai import OpenAI
from serpapi import GoogleSearch
from supabase import create_client, Client

# ----------------- Load ENV -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print("⚠️ Failed to init Supabase:", e)

# ----------------- Logger -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("buywise-rag")

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-IN,en;q=0.9"}
INDIAN_ECOM_SITES = ["flipkart.com", "amazon.in", "croma.com", "reliancedigital.in"]

# --- Memory Store ---
docs: List[Dict] = []
embeddings: List[List[float]] = []
nn_model: Optional[NearestNeighbors] = None

# ----------------- Embedding Cache -----------------
embedding_cache: Dict[str, List[float]] = {}


def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_embedding(text: str) -> List[float]:
    """Cached OpenAI embeddings"""
    if not text:
        raise ValueError("Empty text for embedding")
    key = cache_key(text)
    if key in embedding_cache:
        return embedding_cache[key]
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    emb = resp.data[0].embedding
    embedding_cache[key] = emb
    return emb


# ----------------- Helpers -----------------
def serpapi_search(query: str, site: str = None, num: int = 5) -> List[Dict[str, Any]]:
    if not SERPAPI_API_KEY:
        return []
    q = query if not site else f"{query} site:{site}"
    params = {"q": q, "hl": "en", "gl": "in", "num": num, "api_key": SERPAPI_API_KEY}
    try:
        results = GoogleSearch(params).get_dict()
        return results.get("organic_results", []) or []
    except Exception as e:
        logger.error("SerpAPI failed: %s", e)
        return []


def fetch_page_text(url: str, timeout: int = 5) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if not resp.ok:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return soup.get_text(separator=" ", strip=True)[:20000]
    except Exception as e:
        logger.warning("Fetch page failed: %s", e)
        return ""


def extract_price(text: str) -> int:
    if not text:
        return -1
    patterns = [r"₹\s?([\d,]+)", r"Rs\.?\s?([\d,]+)", r"INR\s?([\d,]+)"]
    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return -1


def upsert_product_doc(url: str, title: str, price: int, snippet: str):
    try:
        emb = get_embedding(snippet)
    except Exception as e:
        logger.warning("Embedding failed for %s: %s", url, e)
        return
    embeddings.append(emb)
    docs.append({"url": url, "title": title, "price": price, "snippet": snippet})

    global nn_model
    if embeddings:
        X = np.array(embeddings, dtype="float32")
        n_neighbors = min(5, len(X))
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn_model.fit(X)


def retrieve_similar(query: str, top_k: int = 5):
    if not nn_model or not docs:
        return []
    try:
        q_emb = np.array([get_embedding(query)], dtype="float32")
        k = min(top_k, len(docs))
        distances, indices = nn_model.kneighbors(q_emb, n_neighbors=k)
        return [
            {
                "document": docs[idx]["snippet"],
                "metadata": docs[idx],
                "distance": float(distances[0][i]),
            }
            for i, idx in enumerate(indices[0])
        ]
    except Exception as e:
        logger.warning("Similarity search failed: %s", e)
        return []


def synthesize_recommendation_with_preferences(user_query, retrieved_docs, radio_selection, slider_values):
    system_msg = (
        "You are a friendly expert on mobiles in India. "
        "Use retrieved snippets, radio choice, and slider values. "
        "Reply in JSON with keys: top_picks (list), quick_summary (list)."
    )
    sources_text = "\n".join(
        f"TITLE: {d['metadata']['title']}\nPRICE: {d['metadata']['price']}\nURL: {d['metadata']['url']}"
        for d in retrieved_docs
    )
    user_msg = (
        f"User query: {user_query}\nRadio: {radio_selection}\nSlider: {slider_values}\n\nSources:\n{sources_text}\n\nJSON only."
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2,
            max_tokens=400,
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error("Chat completion failed: %s", e)
        return {"top_picks": [], "quick_summary": [], "error": str(e)}


# ----------------- Flask -----------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")

# ✅ Proper CORS setup
allowed_origins = [
    "https://buywise-smart-shopper.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    supports_credentials=True
)


@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Unhandled exception: %s", e)
    return jsonify({
        "success": False,
        "data": None,
        "error": "Server error"
    }), 500


# ----------------- Routes -----------------
def process_result(query: str, site: str):
    """Fetch SerpAPI results for a site, scrape pages, and build docs."""
    collected = []
    try:
        for r in serpapi_search(query, site=site, num=2):
            url, title, snippet = r.get("link"), r.get("title", ""), r.get("snippet", "")
            if not url:
                continue
            text = fetch_page_text(url)
            price = extract_price(snippet or text)
            doc = f"{title}\nPrice: {price if price!=-1 else 'Unknown'}\n{snippet or text[:400]}"

            # Dedup by URL
            if any(d["url"] == url for d in docs):
                continue

            upsert_product_doc(url, title, price, doc)
            collected.append({"url": url, "title": title, "price": price, "snippet": doc})
    except Exception as e:
        logger.warning("process_result failed for %s: %s", site, e)
    return collected


@app.route("/search", methods=["POST", "GET"])
def search():
    if request.method == "GET":
        return jsonify({
            "success": True,
            "data": {"status": "ok"},
            "error": None
        })

    payload = request.get_json(force=True, silent=True) or {}
    query = payload.get("query")
    if not query:
        return jsonify({
            "success": False,
            "data": None,
            "error": "Missing query"
        }), 400

    radio_selection = payload.get("radio_selection")
    slider_values = payload.get("slider_values")

    collected = []

    # ⚡ Run sites in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_result, query, site) for site in INDIAN_ECOM_SITES]
        for f in as_completed(futures):
            collected.extend(f.result())

    # ⚡ Limit number of docs per request
    collected = collected[:6]

    retrieved = retrieve_similar(query, top_k=5)
    synthesis = synthesize_recommendation_with_preferences(query, retrieved, radio_selection, slider_values)

    return jsonify({
        "success": True,
        "data": {
            "query": query,
            "radio_selection": radio_selection,
            "slider_values": slider_values,
            "quick_hits": collected,
            "recommendations": synthesis
        },
        "error": None
    })


@app.route("/recommendations", methods=["POST"])
def recommendations():
    return search()


# ----------------- Run -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
