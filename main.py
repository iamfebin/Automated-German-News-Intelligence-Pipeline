import os
import sys
import json
import logging
import warnings
import time
import functools
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import faiss
import requests
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_index import load_index_and_metadata, query_vector_search
from src.nlp_pipeline import merge_subword_entities
from src.drift_monitor import DRIFT_METRICS_FILENAME, DATA_DIR

# Retrieve credentials
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL_NAME}/pipeline/feature-extraction"
TRANSLATE_API_URL = "https://router.huggingface.co/hf-inference/models/Helsinki-NLP/opus-mt-de-en"

# Database state holders
df_meta = pd.DataFrame()
faiss_index = None
local_transformer = None

def init_db():
    global df_meta, faiss_index
    logger.info("Initializing search database and FAISS index...")
    try:
        df_meta, faiss_index = load_index_and_metadata(repo_id=HF_REPO_ID, token=HF_TOKEN)
        logger.info(f"Database successfully loaded. Contains {len(df_meta)} articles.")
    except Exception as e:
        logger.error(f"Error loading database and index: {e}")
        df_meta = pd.DataFrame()
        faiss_index = None

# Initialize FastAPI App
app = FastAPI(title="Multilingual German News Intelligence", description="FastAPI Backend for news search")

@app.on_event("startup")
def startup_event():
    init_db()

# Request schemas
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.35

def get_local_embedding_model():
    global local_transformer
    if local_transformer is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local sentence transformer model for fallback...")
        local_transformer = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return local_transformer

def vectorize_query(query_text: str) -> np.ndarray:
    """
    Vectorizes query using HF Serverless API, falling back to local model if needed.
    """
    query_text = query_text.strip()
    if not query_text:
        raise ValueError("Query text cannot be empty")
        
    if HF_TOKEN:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "x-wait-for-model": "true"
        }
        for attempt in range(3):
            try:
                response = requests.post(API_URL, headers=headers, json={"inputs": query_text}, timeout=15)
                if response.status_code == 200:
                    res_json = response.json()
                    if isinstance(res_json, list):
                        emb = np.array(res_json, dtype=np.float32)
                        emb = emb / np.linalg.norm(emb)
                        return emb
                else:
                    logger.warning(f"HF Inference API status {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"HF Serverless Inference attempt {attempt + 1} failed: {e}")
            time.sleep(1)
            
    # Fallback
    local_model = get_local_embedding_model()
    emb = local_model.encode([query_text], convert_to_numpy=True)[0]
    emb = emb / np.linalg.norm(emb)
    return emb

@functools.lru_cache(maxsize=1024)
def translate_text_cached(text: str) -> str:
    """
    Translates first 3 sentences / lead text of the article JIT using Serverless HF API.
    Utilizes LRU caching to avoid repeating API calls.
    """
    if not text:
        return ""
        
    sentences = text.split(". ")
    lead_text = ". ".join(sentences[:3])
    if len(lead_text) > 600:
        lead_text = lead_text[:600]
    if not lead_text.endswith("."):
        lead_text += "."
        
    if HF_TOKEN:
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "x-wait-for-model": "true"
        }
        for attempt in range(3):
            try:
                response = requests.post(TRANSLATE_API_URL, headers=headers, json={"inputs": lead_text}, timeout=15)
                if response.status_code == 200:
                    res_json = response.json()
                    if isinstance(res_json, list) and len(res_json) > 0:
                        translation = res_json[0].get("translation_text", "").strip()
                        if translation:
                            return translation
                else:
                    logger.warning(f"HF Translation API status {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"HF translation attempt {attempt + 1} failed: {e}")
            time.sleep(1)
            
    return "English translation unavailable (configure HF_TOKEN or check API connection)."

# API Routes
@app.get("/")
def read_root():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/api/status")
def get_status():
    global df_meta
    
    # Calculate source breakdown
    source_counts = {}
    if not df_meta.empty and "source" in df_meta.columns:
        source_counts = df_meta["source"].value_counts().to_dict()
        
    # Load drift metrics
    metrics_path = os.path.join(DATA_DIR, DRIFT_METRICS_FILENAME)
    drift_metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                drift_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading drift metrics: {e}")
            
    return JSONResponse(content={
        "article_count": len(df_meta) if not df_meta.empty else 0,
        "source_breakdown": source_counts,
        "hf_repo_id": HF_REPO_ID or "Local Offline Mode",
        "drift_metrics": drift_metrics
    })

@app.post("/api/search")
def search(request: SearchRequest):
    global df_meta, faiss_index
    
    if df_meta.empty or faiss_index is None:
        return JSONResponse(content={"results": [], "error": "Database not initialized. Please run the ingestion pipeline or sync."})
        
    try:
        # 1. Vectorize Query
        query_vector = vectorize_query(request.query)
        
        # 2. Search FAISS Index
        raw_results = query_vector_search(query_vector, df_meta, faiss_index, top_k=request.top_k)
        
        # 3. Filter and Format
        filtered = []
        for res in raw_results:
            if res["similarity_score"] >= request.threshold:
                # JIT Translation
                summary_en = translate_text_cached(res["body_de"])
                
                # Merge entities on-the-fly
                entities_raw = res.get("entities", [])
                cleaned_entities = merge_subword_entities(entities_raw)
                
                filtered.append({
                    "article_id": res["article_id"],
                    "timestamp": res["timestamp"],
                    "source": res["source"],
                    "url": res["url"],
                    "title_de": res["title_de"],
                    "body_de": res["body_de"],
                    "summary_en": summary_en,
                    "entities": cleaned_entities,
                    "similarity_score": res["similarity_score"]
                })
        return JSONResponse(content={"results": filtered})
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync")
def sync_db():
    logger.info("Manual database sync requested...")
    init_db()
    return JSONResponse(content={"status": "success", "article_count": len(df_meta) if not df_meta.empty else 0})

# Serve static files and data files (like text_drift_report.html)
# We place these mounts after API routes to avoid matching overrides
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
