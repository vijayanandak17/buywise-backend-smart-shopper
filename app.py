import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify, redirect, session
from flask_cors import CORS

# New OpenAI v1+ client
from openai import OpenAI

from serpapi import GoogleSearch  # pip install google-search-results
from supabase import create_client, Client  # pip install supabase

# ----------------- Load ENV -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# initialize new OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# initialize supabase client (ensure SUPABASE_URL and SUPABASE_KEY are set in Render env)
if SUPABASE_URL and SUPABASE_KEY:
    supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    # We'll still let the app run; supabase operations will be no-ops with logged warnings.

# ----------------- Logger -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("buywise-rag")

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-IN,en;q=0.9"}
INDIAN_ECOM_SITES = ["flipkart.com", "amazon.in", "croma.com", "reliancedigital.in"]

# --- Storage and NearestNeighbors setup ---
docs: List[Dict] = []
embeddings: List[List[float]] = []
nn_model: Optional[NearestNeighbors] = None

# ----------------- Helper Functions -----------------

def serpapi_search(query: str, site: str = None, num: int = 5) -> List[Dict[str, Any]]:
    if not SERPAPI_API_KEY:
        logger.warning("SERPAPI_API_KEY missing; skipping serpapi search.")
        return []
    q = query if not site else f"{query} site:{site}"
    params = {"q": q, "hl": "en", "gl": "in", "num": num, "api_key": SERPAPI_API_KEY}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", []) or []
    except Exception as e:
        logger.error("SerpAPI failed: %s", e, exc_info=True)
        return []


def fetch_page_text(url: str, timeout: int = 8) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return soup.get_text(separator=" ", strip=True)[:20000]
    except Exception as e:
        logger.debug("Failed to fetch page text for %s: %s", url, e)
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


def get_embedding(text: str) -> List[float]:
    """
    Use the new OpenAI client to get embeddings.
    Returns embedding vector or raises RuntimeError if not available.
    """
    if not text:
        raise ValueError("Empty text for embedding")

    try:
        # new client call
        resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        # resp expected to have .data[0].embedding or dict access
        embedding = None
        if hasattr(resp, "data"):
            embedding = resp.data[0].embedding
        elif isinstance(resp, dict):
            embedding = resp["data"][0]["embedding"]
        if not embedding:
            raise RuntimeError("No embedding returned")
        return list(embedding)
    except Exception as e:
        logger.error("OpenAI embedding failed: %s", e, exc_info=True)
        # Re-raise so callers can handle; for robustness, callers may choose fallback
        raise


def upsert_product_doc(url: str, title: str, price: int, snippet: str):
    """Add product doc into memory and retrain NN index (keeps up to memory)"""
    try:
        emb = get_embedding(snippet)
    except Exception:
        # If embedding fails, skip this doc but log it
        logger.warning("Skipping upsert for %s due to embedding failure", url)
        return

    embeddings.append(emb)
    docs.append({"url": url, "title": title, "price": price, "snippet": snippet})

    global nn_model
    X = np.array(embeddings, dtype="float32")
    # n_neighbors cannot be 0
    n_neighbors = min(5, len(X))
    if n_neighbors <= 0:
        nn_model = None
        return
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(X)
    logger.debug("NN index updated; docs=%d", len(docs))


def retrieve_similar(query: str, top_k: int = 5):
    if not nn_model or len(docs) == 0:
        return []
    try:
        q_emb = np.array([get_embedding(query)], dtype="float32")
    except Exception:
        logger.warning("Embedding for query failed; returning no similar docs")
        return []

    k = min(top_k, len(docs))
    distances, indices = nn_model.kneighbors(q_emb, n_neighbors=k)
    results = []
    for i, idx in enumerate(indices[0]):
        doc = docs[idx]
        results.append({
            "document": doc["snippet"],
            "metadata": {"url": doc["url"], "title": doc["title"], "price": doc["price"]},
            "distance": float(distances[0][i])
        })
    return results


def synthesize_recommendation_with_preferences(user_query: str,
                                               retrieved_docs: List[Dict[str, Any]],
                                               radio_selection: str,
                                               slider_values: Any):
    # System prompt
    system_msg = (
        "You are a friendly, expert electronics advisor for mobile phones in India. "
        "Use the retrieved product snippets, the user’s radio button choice, and slider values "
        "to suggest the best mobiles. Respond in JSON with keys: top_picks (list) and quick_summary (list)."
    )

    sources_text = ""
    for doc in retrieved_docs:
        md = doc["metadata"]
        sources_text += f"TITLE: {md.get('title','')}\nPRICE: {md.get('price','')}\nURL: {md.get('url','')}\n---\n"

    preferences = f"Radio choice: {radio_selection}\nSlider values: {slider_values}"

    user_msg = (
        f"User query: {user_query}\n"
        f"User preferences:\n{preferences}\n\n"
        f"Sources:\n{sources_text}\n\nJSON only."
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=400
        )

        # Try robust extraction
        content = None
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            # choices[0].message.content for object-like
            try:
                content = resp.choices[0].message.content
            except Exception:
                try:
                    content = resp.choices[0]["message"]["content"]
                except Exception:
                    content = None
        elif isinstance(resp, dict):
            content = resp["choices"][0]["message"]["content"]

        if not content:
            logger.warning("No content from chat completion; returning raw response")
            return {"raw": str(resp)}

        # The model was instructed to return JSON only. Parse safely.
        try:
            parsed = json.loads(content)
            return parsed
        except Exception:
            # If model didn't return strict JSON, return raw string under key
            return {"raw": content}

    except Exception as e:
        logger.error("OpenAI chat completion failed: %s", e, exc_info=True)
        return {"error": "Failed to synthesize recommendations", "details": str(e)}


# ----------------- Flask App -----------------

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")   # ⚠️ use env in prod

# Allow the Vercel front-end and Render backend origin for CORS
CORS(app, resources={r"/*": {"origins": ["https://buywise-smart-shopper.vercel.app", "https://buywise-backend-smart-shopper-1.onrender.com"]}})

# ----------------- Auth Routes -----------------

@app.route("/login/google")
def google_login():
    if supabase is None:
        return jsonify({"error": "Supabase client not configured on server"}), 500
    redirect_url = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": "http://localhost:5000/auth/callback"}
    })
    return redirect(redirect_url.url)


@app.route("/auth/callback")
def auth_callback():
    if supabase is None:
        return jsonify({"error": "Supabase client not configured on server"}), 500
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Missing code param"}), 400
    res = supabase.auth.exchange_code_for_session(code)
    # NOTE: supabase response shape may vary; guard defensively
    user = getattr(res, "user", None) or (res.get("user") if isinstance(res, dict) else None)
    if user:
        user_id = getattr(user, "id", None) or user.get("id")
        email = getattr(user, "email", None) or user.get("email")
        try:
            supabase.table("profiles").upsert({"id": user_id, "email": email}).execute()
        except Exception as e:
            logger.warning("Failed to upsert profile to supabase: %s", e)
        session["user"] = user_id
        return jsonify({"message": "Google login successful", "user": user_id, "email": email})
    return jsonify({"error": "Login failed"}), 400


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})


# ----------------- Search Route -----------------

@app.route("/search", methods=["POST", "GET"])
def search():
    # Accept GET for quick health checks
    if request.method == "GET":
        return jsonify({"status": "ok", "message": "Send POST with JSON payload to run search"}), 200

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    query = payload.get("query")
    radio_selection = payload.get("radio_selection")
    slider_values = payload.get("slider_values")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    # Save search to supabase if configured and user in session
    if supabase and "user" in session:
        try:
            supabase.table("searched_mobiles").insert({
                "user_id": session["user"],
                "mobile_name": query,
                "radio_selection": radio_selection,
                "slider_values": json.dumps(slider_values) if slider_values else None
            }).execute()
        except Exception as e:
            logger.warning("Failed to insert search record to supabase: %s", e)

    sites = payload.get("sites") or INDIAN_ECOM_SITES
    collected = []

    # Crawl a few results per site (be mindful of rate limits)
    for site in sites:
        for r in serpapi_search(query, site=site, num=3):
            url = r.get("link")
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            if not url:
                continue
            text = fetch_page_text(url)
            price = extract_price(snippet or text)
            short_doc = f"{title}\nPrice: {price if price!=-1 else 'Unknown'}\n{snippet or text[:500]}"
            upsert_product_doc(url, title, price, short_doc)
            collected.append({"url": url, "title": title, "price": price, "snippet": short_doc})

    # Retrieve similar docs using NN index
    retrieved = retrieve_similar(query, top_k=5)

    # Ask the model to synthesize recommendations (non-blocking behavior could be added later)
    synthesis = synthesize_recommendation_with_preferences(query, retrieved, radio_selection, slider_values)

    return jsonify({
        "query": query,
        "radio_selection": radio_selection,
        "slider_values": slider_values,
        "quick_hits": collected[:5],
        "recommendations": synthesis
    })


# ----------------- Recommendations Route -----------------

@app.route("/recommendations", methods=["POST"])
def recommendations():
    """Frontend calls this instead of /search"""
    return search()


# ----------------- Run -----------------
if __name__ == "__main__":
    # Respect runtime PORT env variable used by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
