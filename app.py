import os
import sys
import json
import logging
import warnings
from typing import Tuple, Optional, List, Dict

# Suppress SyntaxWarnings from third-party libraries (like evidently under Python 3.12+)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import requests
import time

# Set page configuration first
st.set_page_config(
    page_title="Multilingual German News Intelligence",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_index import load_index_and_metadata, query_vector_search
from src.drift_monitor import DRIFT_REPORT_FILENAME, DRIFT_METRICS_FILENAME, DATA_DIR

# Safe loading of Streamlit secrets to prevent crash if secrets.toml is missing
try:
    HF_TOKEN = st.secrets.get("HF_TOKEN") or st.secrets.get("HF_WRITE_TOKEN")
    HF_REPO_ID = st.secrets.get("HF_REPO_ID")
except Exception:
    HF_TOKEN = None
    HF_REPO_ID = None

# Fallback to environment variables
HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN")
HF_REPO_ID = HF_REPO_ID or os.environ.get("HF_REPO_ID")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL_NAME}"

# Custom CSS for Premium Design Look (dark mode friendly, glassmorphism, nice badges)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    
    .news-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #334155;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .news-card:hover {
        transform: translateY(-2px);
        border-color: #475569;
    }
    
    .news-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    
    .news-source {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .news-date {
        color: #9ca3af;
        font-size: 0.85rem;
    }
    
    .news-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 10px;
        color: #f8fafc;
    }
    
    .news-url-link {
        font-size: 0.85rem;
        color: #60a5fa;
        text-decoration: none;
    }
    
    .news-url-link:hover {
        text-decoration: underline;
    }
    
    .summary-box {
        background-color: #0f172a;
        border-left: 4px solid #10b981;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 16px 0;
        color: #e2e8f0;
        font-style: italic;
    }
    
    .entity-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 15px;
    }
    
    .badge {
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        display: flex;
        align-items: center;
    }
    
    .badge-per { background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-loc { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-org { background-color: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    
    .drift-metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        border: 1px solid #334155;
        padding: 20px;
        text-align: center;
    }
    
    .drift-metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 1. Cache the loading of Metadata and FAISS Index (15-minute TTL to pick up new uploads)
@st.cache_resource(show_spinner="Loading semantic database index...", ttl=900)
def get_cached_database(repo_id: Optional[str], token: Optional[str]) -> Tuple[pd.DataFrame, Optional[object]]:
    return load_index_and_metadata(repo_id=repo_id, token=token)

df, index = get_cached_database(HF_REPO_ID, HF_TOKEN)

# 2. Cached local embedding model for fallback
@st.cache_resource
def get_local_embedding_model():
    from sentence_transformers import SentenceTransformer
    logger.info("Loading local sentence transformer model for fallback...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def vectorize_query(query_text: str, token: Optional[str]) -> np.ndarray:
    """
    Vectorizes search query using HF Serverless Inference API,
    falling back to local CPU sentence transformer if token is missing or API errors.
    """
    # Normalize query text
    query_text = query_text.strip()
    
    # Try Serverless HF API if token is provided
    if token:
        headers = {"Authorization": f"Bearer {token}"}
        for attempt in range(3):
            try:
                response = requests.post(API_URL, headers=headers, json={"inputs": query_text}, timeout=10)
                if response.status_code == 200:
                    res_json = response.json()
                    if isinstance(res_json, list):
                        emb = np.array(res_json, dtype=np.float32)
                        # Normalize vector
                        emb = emb / np.linalg.norm(emb)
                        return emb
                else:
                    logger.warning(f"HF API returned status {response.status_code} (attempt {attempt + 1}): {response.text}")
            except Exception as e:
                logger.warning(f"Hugging Face Serverless Inference attempt {attempt + 1} failed: {e}")
            
            if attempt < 2:
                time.sleep(1)
            else:
                logger.error("Hugging Face Serverless Inference failed after 3 attempts. Falling back to local model.")
            
    # Fallback to local model execution
    local_model = get_local_embedding_model()
    emb = local_model.encode([query_text], convert_to_numpy=True)[0]
    emb = emb / np.linalg.norm(emb)
    return emb

@st.cache_data(show_spinner="Translating article summary...", ttl=3600)
def translate_text_cached(text: str, token: Optional[str]) -> str:
    """
    Translates first 3 sentences / lead text of the article JIT
    using the HF Serverless Inference API for Helsinki-NLP/opus-mt-de-en.
    Falls back gracefully if the API fails or token is missing.
    """
    if not text:
        return ""
    
    # Extract lead content (first 3 sentences or up to ~600 chars)
    sentences = text.split(". ")
    lead_text = ". ".join(sentences[:3])
    if len(lead_text) > 600:
        lead_text = lead_text[:600]
    if not lead_text.endswith("."):
        lead_text += "."
        
    if token:
        headers = {"Authorization": f"Bearer {token}"}
        api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-de-en"
        for attempt in range(3):
            try:
                response = requests.post(api_url, headers=headers, json={"inputs": lead_text}, timeout=10)
                if response.status_code == 200:
                    res_json = response.json()
                    if isinstance(res_json, list) and len(res_json) > 0:
                        translation = res_json[0].get("translation_text", "").strip()
                        if translation:
                            return translation
                else:
                    logger.warning(f"HF translation API returned status {response.status_code} (attempt {attempt + 1}): {response.text}")
            except Exception as e:
                logger.warning(f"Hugging Face Serverless translation attempt {attempt + 1} failed: {e}")
            
            if attempt < 2:
                time.sleep(1)
            else:
                logger.error("Hugging Face Serverless translation failed after 3 attempts.")
            
    return "English translation unavailable (configure HF_TOKEN or check API connection)."

# Layout setup
st.markdown("<h1 class='main-title'>📰 German News Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Semantic search and localized intelligence across regional German news feeds (Tagesschau, DW, Spiegel)</p>", unsafe_allow_html=True)

# Sidebar configurations
with st.sidebar:
    st.markdown("### Platform Configurations")
    if HF_REPO_ID:
        st.success(f"Connected to Hugging Face Hub Dataset:\n`{HF_REPO_ID}`")
        if st.button("🔄 Sync & Refresh Database", help="Clears cache and forces a redownload of the latest news database from Hugging Face."):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.info("Running in local mode. No Hugging Face dataset repository configured.")
        
    st.markdown("---")
    st.markdown("### Total Articles Indexed")
    if not df.empty:
        st.metric(label="Articles Ingested", value=len(df))
        # Outlets breakdown
        sources = df["source"].value_counts()
        for src, count in sources.items():
            st.write(f"- **{src}**: {count} articles")
    else:
        st.metric(label="Articles Ingested", value=0)
        
    st.markdown("---")
    st.markdown("### Developer Context")
    st.markdown(
        "This system uses a zero-cost serverless NLP architecture. "
        "Scraping, embedding generation, NER, and Evidently AI drift reports are pre-computed in daily GitHub Actions batches. "
        "Inference is offloaded to Hugging Face Serverless APIs to maintain Streamlit's 1GB RAM container budget."
    )

# Tab Layout
tab1, tab2 = st.tabs(["🔍 Semantic Search Portal", "🩺 MLOps Pipeline Health"])

# TAB 1: SEARCH PORTAL
with tab1:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input(
            label="Search Query",
            placeholder="Type your search in English or German (e.g., 'Scholz meeting with French President' or 'Energiekrise')...",
            label_visibility="collapsed"
        )
    with col2:
        top_k = st.slider("Top Results", min_value=1, max_value=20, value=5)
    with col3:
        similarity_threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help=(
                "Minimum cosine similarity score required. Recommended: 0.35 - 0.45.\n\n"
                "• 0.60+: Very strong relevance\n"
                "• 0.40 - 0.60: Good relevance\n"
                "• < 0.30: Mostly unrelated noise\n\n"
                "Note: A threshold above 0.80 is extremely restrictive and will filter out almost all matches."
            )
        )

    if query:
        if df.empty or index is None or index.ntotal == 0:
            st.warning("No data found in the database. Please run the ETL ingestion pipeline first to scrape and index news articles.")
        else:
            with st.spinner("Embedding query and matching vector space..."):
                query_emb = vectorize_query(query, HF_TOKEN)
                results = query_vector_search(query_emb, df, index, top_k=top_k)
                
            if not results:
                st.info("No matching articles found.")
            else:
                # Filter results by similarity threshold
                filtered_results = [res for res in results if res["similarity_score"] >= similarity_threshold]
                
                if not filtered_results:
                    max_score = results[0]["similarity_score"]
                    max_title = results[0]["title_de"]
                    st.warning(
                        f"No articles matched your Relevance Threshold of **{similarity_threshold:.2f}**.\n\n"
                        f"The closest match found was **'{max_title}'** with a similarity score of **{max_score:.4f}**.\n\n"
                        f"Try lowering the threshold or searching for different terms."
                    )
                else:
                    st.markdown(f"#### Found {len(filtered_results)} contextually relevant articles:")
                    
                    for res in filtered_results:
                        # NER badge tags
                        badge_html = ""
                        entities = res["entities"]
                        seen_badges = set()
                        
                        for ent in entities:
                            word = ent["word"].strip()
                            etype = ent["entity"].lower()
                            # Deduplicate entities
                            badge_key = f"{word}||{etype}"
                            if badge_key in seen_badges:
                                continue
                            seen_badges.add(badge_key)
                            
                            b_class = "badge-per" if etype == "per" else "badge-loc" if etype == "loc" else "badge-org"
                            badge_html += f"<span class='badge {b_class}'>{etype.upper()}: {word}</span>"
                            
                        # Source badge color
                        src = res["source"]
                        
                        # Translate JIT
                        summary_en = translate_text_cached(res["body_de"], HF_TOKEN)
                        
                        # Custom card HTML
                        st.markdown(f"""
                        <div class='news-card'>
                            <div class='news-header'>
                                <span class='news-source'>{src}</span>
                                <span class='news-date'>{res['timestamp']} | Match Score: {res['similarity_score']:.4f}</span>
                            </div>
                            <div class='news-title'>{res['title_de']}</div>
                            <div class='summary-box'>
                                <strong>Bilingual English Summary:</strong><br>
                                {summary_en}
                            </div>
                            <div style='margin-bottom: 10px;'>
                                <a class='news-url-link' href='{res['url']}' target='_blank'>Read original article page ↗</a>
                            </div>
                            <div class='entity-container'>
                                {badge_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Original text collapsible segment
                        with st.expander("Show Original German Text"):
                            st.write(res["body_de"])
    else:
        st.info("Enter a query above to explore regional German news articles semantically.")

# TAB 2: MLOPS DATA DRIFT HEALTH
with tab2:
    st.markdown("### MLOps Ingestion Pipeline Health & Data Drift Diagnostics")
    st.markdown(
        "Linguistic and semantic profile distributions shift as global news topics change. "
        "The metrics below display drift comparisons between today's ingestion batch and a rolling reference baseline of the past 14 days."
    )
    
    # 1. Load drift metrics JSON
    metrics_path = os.path.join(DATA_DIR, DRIFT_METRICS_FILENAME)
    metrics_loaded = False
    drift_metrics = {}
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                drift_metrics = json.load(f)
            metrics_loaded = True
        except Exception as e:
            logger.error(f"Error loading drift metrics JSON: {e}")
            
    if metrics_loaded:
        emb_drift = drift_metrics.get("embedding_drift", {})
        psi = emb_drift.get("population_stability_index", 0.0)
        wd = emb_drift.get("wasserstein_distance", 0.0)
        status = emb_drift.get("status", "Unknown")
        ref_count = drift_metrics.get("reference_count", 0)
        cur_count = drift_metrics.get("current_count", 0)
        
        # Color coordinate status
        status_color = "#10b981" # Green
        if status == "Significant Drift":
            status_color = "#ef4444" # Red
        elif status == "Moderate Drift":
            status_color = "#f59e0b" # Yellow
            
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class='drift-metric-card'>
                <div style='font-size:0.9rem; color:#9ca3af;'>Embedding PSI</div>
                <div class='drift-metric-value'>{psi:.4f}</div>
                <div style='font-size:0.85rem; color:#9ca3af;'>Population Stability Index</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class='drift-metric-card'>
                <div style='font-size:0.9rem; color:#9ca3af;'>Wasserstein Distance</div>
                <div class='drift-metric-value'>{wd:.4f}</div>
                <div style='font-size:0.85rem; color:#9ca3af;'>Earth Mover's Distance</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class='drift-metric-card'>
                <div style='font-size:0.9rem; color:#9ca3af;'>Pipeline Status</div>
                <div class='drift-metric-value' style='color:{status_color};'>{status}</div>
                <div style='font-size:0.85rem; color:#9ca3af;'>Semantic Drift Threshold</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class='drift-metric-card'>
                <div style='font-size:0.9rem; color:#9ca3af;'>Sample Breakdown</div>
                <div class='drift-metric-value'>{cur_count} / {ref_count}</div>
                <div style='font-size:0.85rem; color:#9ca3af;'>Validation / Reference Sizes</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown(f"<p style='color:#9ca3af; font-size:0.85rem; margin-top:10px;'>Drift report last updated: {drift_metrics.get('timestamp', 'Unknown')}</p>", unsafe_allow_html=True)
    else:
        st.info("Drift diagnostics metrics not yet generated. Please execute the ETL pipeline to view.")
        
    st.markdown("---")
    
    # 2. Render Evidently AI HTML Report
    report_path = os.path.join(DATA_DIR, DRIFT_REPORT_FILENAME)
    if os.path.exists(report_path):
        st.markdown("#### Detailed Evidently AI Text Drift Diagnostics")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report_html = f.read()
            components.html(report_html, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Error rendering Evidently AI HTML report: {e}")
    else:
        st.info("Evidently AI detailed report not yet generated. Please execute the ETL pipeline to view.")
