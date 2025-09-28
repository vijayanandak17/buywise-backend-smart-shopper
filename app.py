"""
buywise-flask-rag/app.py

Flask RAG app with:
- scikit-learn NearestNeighbors (Windows-friendly)
- CORS enabled for frontend
- Supabase Google login (only)
- Storing user searches in Supabase
- Accepting frontend inputs (radio buttons + sliders) from Lovable.dev
- Responding at /recommendations (POST)
"""

import os
import re
import json
import logging
from typing import List, Dict, Any

import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify, redirect, session
from flask_cors import CORS

import openai
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

openai.api_key = OPENAI_API_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------- Logger -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("buywise-rag")

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-IN,en;q=0.9"}
INDIAN_ECOM_SITES = ["flipkart.com", "amazon.in", "croma.com", "reliancedigital.in"]

# --- Storage and NearestNeighbors setup ---
docs: List[Dict] = []
embeddings: List[List[float]] = []
nn_model: NearestNeighbors = None

# ----------------- Helper Functions -----------------

def serpapi_search(query: str, site: str = None, num: int = 5) -> List[Dict[str, Any]]:
    q = query if not site else f"{query} site:{site}"
    params = {"q": q, "hl": "en", "gl": "in", "num": num, "api_key": SERPAPI_API_KEY}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", []) or []
    except Exception as e:
        logger.error("SerpAPI failed: %s", e)
        return []


def fetch_page_text(url: str, timeout: int = 8) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return soup.get_text(separator=" ", strip=True)[:20000]
    except Exception:
        return ""


def extract_price(text: str) -> int:
    patterns = [r"₹\s?([\d,]+)", r"Rs\.?\s?([\d,]+)", r"INR\s?([\d,]+)"]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1).replace(",", ""))
    return -1


def get_embedding(text: str) -> List[float]:
    resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return resp["data"][0]["embedding"]


def upsert_product_doc(url: str, title: str, price: int, snippet: str):
    """Add product doc into memory and retrain NN index"""
    emb = get_embedding(snippet)
    embeddings.append(emb)
    docs.append({"url": url, "title": title, "price": price, "snippet": snippet})

    global nn_model
    X = np.array(embeddings, dtype="float32")
    nn_model = NearestNeighbors(n_neighbors=min(5, len(X)), metric="euclidean")
    nn_model.fit(X)


def retrieve_similar(query: str, top_k: int = 5):
    if not nn_model:
        return []
    q_emb = np.array([get_embedding(query)], dtype="float32")
    distances, indices = nn_model.kneighbors(q_emb, n_neighbors=min(top_k, len(docs)))
    results = []
    for i, idx in enumerate(indices[0]):
        doc = docs[idx]
        results.append({
            "document": doc["snippet"],
            "metadata": {"url": doc["url"], "title": doc["title"], "price": doc["price"]},
            "distance": float(distances[0][i])
        })
    return results


def synthesize_recommendation_with_preferences(user_query: str, retrieved_docs: List[Dict[str, Any]], radio_selection: str, slider_values: Any):
    system_msg = (
        "You are a friendly, expert electronics advisor for mobile phones in India. "
        "Use the retrieved product snippets, the user’s radio button choice, and slider values "
        "to suggest the best mobiles. "
        "Respond in JSON with keys: top_picks (list) and quick_summary (list)."
    )

    sources_text = ""
    for doc in retrieved_docs:
        md = doc["metadata"]
        sources_text += f"TITLE: {md['title']}\nPRICE: {md['price']}\nURL: {md['url']}\n---\n"

    preferences = f"Radio choice: {radio_selection}\nSlider values: {slider_values}"

    user_msg = (
        f"User query: {user_query}\n"
        f"User preferences:\n{preferences}\n\n"
        f"Sources:\n{sources_text}\n\nJSON only."
    )

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": user_msg}],
        temperature=0.2,
        max_tokens=400
    )
    try:
        return json.loads(resp["choices"][0]["message"]["content"])
    except Exception:
        return {"raw": resp["choices"][0]["message"]["content"]}

# ----------------- Flask App -----------------

app = Flask(__name__)
app.secret_key = "supersecret"   # ⚠️ use env in prod
CORS(app, resources={r"/*": {"origins": "https://buywise-smart-shopper.vercel.app/"}})

# ----------------- Auth Routes -----------------

@app.route("/login/google")
def google_login():
    redirect_url = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": "http://localhost:5000/auth/callback"}
    })
    return redirect(redirect_url.url)


@app.route("/auth/callback")
def auth_callback():
    code = request.args.get("code")
    res = supabase.auth.exchange_code_for_session(code)
    if res.user:
        user_id = res.user.id
        email = res.user.email
        supabase.table("profiles").upsert({"id": user_id, "email": email}).execute()
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
    payload = request.get_json(force=True)

    query = payload.get("query")
    radio_selection = payload.get("radio_selection")
    slider_values = payload.get("slider_values")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    if "user" in session:
        supabase.table("searched_mobiles").insert({
            "user_id": session["user"],
            "mobile_name": query,
            "radio_selection": radio_selection,
            "slider_values": json.dumps(slider_values) if slider_values else None
        }).execute()

    sites = payload.get("sites") or INDIAN_ECOM_SITES
    collected = []

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

    retrieved = retrieve_similar(query, top_k=5)
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
    app.run(host="0.0.0.0", port=5000, debug=True)
