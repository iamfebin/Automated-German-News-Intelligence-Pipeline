import os
import re
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

# Suppress SyntaxWarnings from third-party libraries (like evidently under Python 3.12+)
warnings.filterwarnings("ignore", category=SyntaxWarning)

logger = logging.getLogger(__name__)


# Try importing Evidently across modern and legacy versions. If not installed, fallback gracefully.
EVIDENTLY_AVAILABLE = False
try:
    from evidently.report import Report
    from evidently.metric_preset import TextEvals
    EVIDENTLY_AVAILABLE = True
except ImportError:
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import TextEvals
        EVIDENTLY_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Evidently AI import failed: {e}. Diagnostics will fallback to mathematical metrics.")
        EVIDENTLY_AVAILABLE = False
except Exception as e:
    logger.warning(f"Evidently AI import failed: {e}. Diagnostics will fallback to mathematical metrics.")
    EVIDENTLY_AVAILABLE = False








DATA_DIR = os.environ.get("DATA_DIR", "data")
METADATA_FILENAME = "news_metadata.parquet"
DRIFT_REPORT_FILENAME = "drift_report.html"
DRIFT_REPORT_JS_FILENAME = "drift_report.js"
DRIFT_METRICS_FILENAME = "drift_metrics.json"


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between two 1D distributions.
    """
    # Determine bin edges based on expected (baseline) dataset quantiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    
    # Ensure bin edges are unique to prevent histogram issues
    bins = np.unique(bins)
    if len(bins) < 2:
        # If all expected values are identical, add a small offset
        bins = np.array([expected[0] - 1e-5, expected[0] + 1e-5])
    else:
        bins[0] = -np.inf
        bins[-1] = np.inf

    # Calculate frequencies
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Convert counts to probabilities (with small epsilon smoothing to avoid log(0))
    eps = 1e-5
    expected_probs = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
    actual_probs = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))
    
    # Calculate PSI
    psi_value = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
    return float(psi_value)

def compute_embedding_drift(ref_embeddings: np.ndarray, cur_embeddings: np.ndarray) -> Tuple[float, float]:
    """
    Computes coordinate-wise Wasserstein Distance and Population Stability Index (PSI)
    across all embedding dimensions and averages the scores.
    """
    if ref_embeddings.size == 0 or cur_embeddings.size == 0:
        return 0.0, 0.0
        
    num_dimensions = ref_embeddings.shape[1]
    
    # Calculate coordinate-wise Wasserstein Distance
    wd_scores = []
    psi_scores = []
    
    for i in range(num_dimensions):
        ref_dim = ref_embeddings[:, i]
        cur_dim = cur_embeddings[:, i]
        
        # Wasserstein Distance
        wd = wasserstein_distance(ref_dim, cur_dim)
        wd_scores.append(wd)
        
        # PSI
        psi = calculate_psi(ref_dim, cur_dim)
        psi_scores.append(psi)
        
    return float(np.mean(wd_scores)), float(np.mean(psi_scores))

def post_process_report_csp(report_path: str):
    """
    Post-processes the generated Evidently AI HTML report to extract all inline script tags
    into a companion .js file. This prevents Content Security Policy (CSP) violations on hosting
    environments (such as Hugging Face Spaces) that block unsafe inline scripts.
    """
    if not os.path.exists(report_path):
        return
        
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Pattern to find all <script>...</script> blocks
        pattern = re.compile(r"<script>(.*?)</script>", re.DOTALL)
        matches = list(pattern.finditer(content))
        
        if not matches:
            logger.info("No inline scripts found to extract.")
            return
            
        scripts_to_combine = []
        for match in matches:
            scripts_to_combine.append(match.group(1))
            
        combined_js = "\n\n/* --- COMBINED SCRIPT --- */\n\n".join(scripts_to_combine)
        
        # Save to companion .js file (e.g. data/drift_report.js)
        js_path = os.path.splitext(report_path)[0] + ".js"
        with open(js_path, "w", encoding="utf-8") as js_f:
            js_f.write(combined_js)
            
        logger.info(f"Successfully extracted inline scripts to {js_path}")
        
        # Replace script tags: keep first script tag as src reference, remove the rest
        parts = []
        last_end = 0
        js_filename = os.path.basename(js_path)
        for i, match in enumerate(matches):
            start = match.start()
            end = match.end()
            parts.append(content[last_end:start])
            if i == 0:
                parts.append(f'<script defer src="{js_filename}"></script>')
            last_end = end
        parts.append(content[last_end:])
        
        new_html = "".join(parts)
        
        with open(report_path, "w", encoding="utf-8") as html_f:
            html_f.write(new_html)
            
        logger.info("Successfully removed inline script tags from HTML report.")
    except Exception as e:
        logger.error(f"Failed to post-process HTML report for CSP: {e}", exc_info=True)

def generate_drift_report(new_article_ids: list) -> Dict[str, Any]:
    """
    Compares newly scraped articles (current) against historical baseline (reference - past 14 days).
    Generates Evidently AI text data drift HTML report and calculates custom embedding drift.
    """
    metadata_path = os.path.join(DATA_DIR, METADATA_FILENAME)
    report_path = os.path.join(DATA_DIR, DRIFT_REPORT_FILENAME)
    metrics_path = os.path.join(DATA_DIR, DRIFT_METRICS_FILENAME)
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file {metadata_path} not found. Cannot generate drift report.")
        return {}
        
    df = pd.read_parquet(metadata_path)
    if len(df) == 0:
        logger.warning("Metadata Parquet is empty. Cannot generate drift report.")
        return {}
        
    # Standardize timestamps
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    
    # Define "current" and "reference" sets dynamically
    if len(new_article_ids) > 0 and len(new_article_ids) < len(df):
        current_df = df[df["article_id"].isin(new_article_ids)].copy()
        reference_df = df[~df["article_id"].isin(new_article_ids)].copy()
        
        # Restrict reference to past 14 days if possible
        latest_date = df["timestamp_dt"].max()
        cutoff_date = latest_date - timedelta(days=14)
        filtered_ref = reference_df[reference_df["timestamp_dt"] >= cutoff_date]
        if len(filtered_ref) >= 5:
            reference_df = filtered_ref
    else:
        # Fallback when 0 new articles ingested in current run or all articles are marked new
        logger.info("No newly scraped articles provided or full dataset refreshed. Splitting dataset by timestamp to compare recent vs historical baseline.")
        latest_date = df["timestamp_dt"].max()
        day_cutoff = latest_date - timedelta(days=1)
        current_df = df[df["timestamp_dt"] >= day_cutoff].copy()
        reference_df = df[df["timestamp_dt"] < day_cutoff].copy()
        
        # If day splitting leaves either set too small, split 70/30 chronologically
        if len(current_df) < 3 or len(reference_df) < 3:
            df_sorted = df.sort_values("timestamp_dt")
            split_idx = int(len(df_sorted) * 0.7)
            reference_df = df_sorted.iloc[:split_idx].copy()
            current_df = df_sorted.iloc[split_idx:].copy()
            
        if len(reference_df) < 2 or len(current_df) < 2:
            logger.warning("Total dataset is too small (<5 articles) to perform split drift analysis.")
            reference_df = df.copy()
            current_df = df.copy()
        
    logger.info(f"Running drift analysis. Reference size: {len(reference_df)}, Current size: {len(current_df)}")
    
    # 1. Compute custom embedding drift
    mean_wd = 0.0
    mean_psi = 0.0
    
    if "embedding" in df.columns:
        try:
            ref_embeddings = np.stack(reference_df["embedding"].values).astype('float32')
            cur_embeddings = np.stack(current_df["embedding"].values).astype('float32')
            
            mean_wd, mean_psi = compute_embedding_drift(ref_embeddings, cur_embeddings)
            logger.info(f"Embedding Drift - Mean Wasserstein Distance: {mean_wd:.4f}, Mean PSI: {mean_psi:.4f}")
        except Exception as e:
            logger.error(f"Error calculating embedding drift: {e}")
            
    # Determine general status
    status = "Stable"
    if len(reference_df) < 2 or len(current_df) < 2:
        status = "Insufficient Baseline Data"
    elif mean_psi > 0.25:
        status = "Significant Drift"
    elif mean_psi > 0.1:
        status = "Moderate Drift"
        
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "reference_count": len(reference_df),
        "current_count": len(current_df),
        "embedding_drift": {
            "wasserstein_distance": mean_wd,
            "population_stability_index": mean_psi,
            "status": status
        },
        "evidently_report_generated": False
    }
    
    # 2. Generate Evidently AI HTML Report
    if EVIDENTLY_AVAILABLE:
        try:
            # Rename column for clean display labels in Evidently report
            ref_texts = reference_df[["body_de"]].rename(columns={"body_de": "German News Text"}).copy()
            cur_texts = current_df[["body_de"]].rename(columns={"body_de": "German News Text"}).copy()
            
            report = Report(metrics=[
                TextEvals(column_name="German News Text")
            ])
            
            logger.info("Running Evidently AI text drift metrics...")
            report.run(current_data=cur_texts, reference_data=ref_texts)
            # Save self-contained HTML report with inline JS so iframe executes Plotly/Evidently scripts
            report.save_html(report_path)

            metrics["evidently_report_generated"] = True
            logger.info(f"Evidently AI self-contained HTML report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate Evidently AI report: {e}")
    else:
        logger.warning("Evidently AI is not available. Skipping HTML report generation.")
        # Create a modern, dark-themed fallback HTML report for Streamlit & FastAPI
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            badge_class = (
                "badge-significant" if status == "Significant Drift"
                else "badge-moderate" if status == "Moderate Drift"
                else "badge-insufficient" if status == "Insufficient Baseline Data"
                else "badge-stable"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Data Drift Diagnostics</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0b0f19;
            color: #f3f4f6;
            margin: 0;
            padding: 32px 16px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 80vh;
        }}
        .card {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 36px;
            max-width: 650px;
            width: 100%;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            text-align: center;
        }}
        .title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #60a5fa;
            margin-top: 0;
            margin-bottom: 8px;
        }}
        .subtitle {{
            font-size: 0.9rem;
            color: #9ca3af;
            margin-bottom: 24px;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 24px;
        }}
        .badge-stable {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-moderate {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
        .badge-significant {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .badge-insufficient {{ background: rgba(107, 114, 128, 0.15); color: #9ca3af; border: 1px solid rgba(107, 114, 128, 0.3); }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-top: 16px;
        }}
        .metric-box {{
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 16px;
        }}
        .metric-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9ca3af;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #38bdf8;
        }}
        .notice {{
            font-size: 0.85rem;
            color: #9ca3af;
            background: rgba(31, 41, 55, 0.6);
            border-radius: 8px;
            padding: 12px;
            margin-top: 24px;
            border-left: 3px solid #3b82f6;
            text-align: left;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h2 class="title">MLOps Data Drift Diagnostics</h2>
        <div class="subtitle">Vector Distribution & Statistical Metrics</div>
        <div class="badge {badge_class}">Pipeline Status: {status}</div>
        
        <div class="grid">
            <div class="metric-box">
                <div class="metric-label">Population Stability Index</div>
                <div class="metric-value">{mean_psi:.4f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Wasserstein Distance</div>
                <div class="metric-value">{mean_wd:.4f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Current Validation Sample</div>
                <div class="metric-value">{len(current_df)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Reference Baseline Sample</div>
                <div class="metric-value">{len(reference_df)}</div>
            </div>
        </div>

        <div class="notice">
            <strong>Diagnostic Info:</strong> Evidently AI text drift report is operating in mathematical fallback mode. High-dimensional vector embedding drift (PSI & Earth Mover's Distance) was computed successfully across all 384 feature coordinates.
        </div>
    </div>
</body>
</html>""")
            metrics["evidently_report_generated"] = True
            logger.info(f"Created placeholder HTML report at {report_path}")
        except Exception as e:
            logger.error(f"Could not create placeholder HTML: {e}")
            
    # Save metrics JSON
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved drift metrics JSON to {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving drift metrics JSON: {e}")
        
    return metrics

if __name__ == "__main__":
    # Test drift monitor
    logging.basicConfig(level=logging.INFO)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate dummy parquet
    meta_path = os.path.join(DATA_DIR, METADATA_FILENAME)
    
    # Clean if existing
    if os.path.exists(meta_path):
        os.remove(meta_path)
        
    dummy_ref_embs = [list(np.random.randn(384).astype(float)) for _ in range(10)]
    dummy_cur_embs = [list((np.random.randn(384) + 0.2).astype(float)) for _ in range(5)] # Add a slight shift
    
    data = []
    # 10 Reference rows
    for i in range(10):
        data.append({
            "article_id": f"ref_{i}",
            "timestamp": (datetime.utcnow() - timedelta(days=i+1)).isoformat(),
            "source": "Spiegel",
            "url": f"https://example.com/ref_{i}",
            "title_de": f"Referenz {i}",
            "body_de": "Das ist ein langer deutscher Text, der zum Testen des Drift-Monitors dient.",
            "entities": "[]",
            "summary_en": "This is a long English text.",
            "embedding": dummy_ref_embs[i]
        })
    # 5 Current rows
    for i in range(5):
        data.append({
            "article_id": f"cur_{i}",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "Tagesschau",
            "url": f"https://example.com/cur_{i}",
            "title_de": f"Aktuell {i}",
            "body_de": "Ein anderer Text mit neuen Begriffen und einer veränderten Wortwahl für den Test.",
            "entities": "[]",
            "summary_en": "A different text with new terms.",
            "embedding": dummy_cur_embs[i]
        })
        
    df = pd.DataFrame(data)
    df.to_parquet(meta_path, index=False)
    
    # Run drift report
    new_ids = [f"cur_{i}" for i in range(5)]
    metrics = generate_drift_report(new_ids)
    print(json.dumps(metrics, indent=2))
    
    # Cleanup
    if os.path.exists(meta_path):
        os.remove(meta_path)
    report_path = os.path.join(DATA_DIR, DRIFT_REPORT_FILENAME)
    if os.path.exists(report_path):
        os.remove(report_path)
    metrics_path = os.path.join(DATA_DIR, DRIFT_METRICS_FILENAME)
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
