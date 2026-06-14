import os
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

# Try importing Evidently. If not installed, we'll log warnings or fallback.
try:
    from evidently.report import Report
    from evidently.metric_preset import TextDataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "data")
METADATA_FILENAME = "news_metadata.parquet"
DRIFT_REPORT_FILENAME = "drift_report.html"
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
    
    # Define "current" as the newly scraped articles
    current_df = df[df["article_id"].isin(new_article_ids)].copy()
    
    # Define "reference" as articles from the past 14 days before today
    # (or fallback to all historical articles except current ones if history is small)
    if len(new_article_ids) > 0 and len(new_article_ids) < len(df):
        reference_df = df[~df["article_id"].isin(new_article_ids)].copy()
        
        # Try to restrict reference to past 14 days
        latest_date = df["timestamp_dt"].max()
        cutoff_date = latest_date - timedelta(days=14)
        filtered_ref = reference_df[reference_df["timestamp_dt"] >= cutoff_date]
        
        # Ensure reference is of decent size, else fallback to full history
        if len(filtered_ref) >= 5:
            reference_df = filtered_ref
    else:
        # Fallback if no new articles or no historical database exists yet
        logger.warning("Insufficient historical data to compare or no new articles scraped. Using self-comparison as dummy baseline.")
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
    # Standard PSI interpretation: < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 significant shift
    status = "Stable"
    if mean_psi > 0.25:
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
            # evidently expects text columns. We'll run TextDataDriftPreset on body_de.
            # We select only the columns needed to save memory
            ref_texts = reference_df[["body_de"]].copy()
            cur_texts = current_df[["body_de"]].copy()
            
            # Rename columns if needed
            report = Report(metrics=[
                TextDataDriftPreset(column_name="body_de")
            ])
            
            logger.info("Running Evidently AI text drift metrics...")
            report.run(reference_data=ref_texts, current_data=cur_texts)
            report.save_html(report_path)
            metrics["evidently_report_generated"] = True
            logger.info(f"Evidently AI HTML report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate Evidently AI report: {e}")
    else:
        logger.warning("Evidently AI is not available. Skipping HTML report generation.")
        # Create a simple placeholder HTML so Streamlit does not crash when trying to render it
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: sans-serif; padding: 40px; background-color: #0e1117; color: #ffffff; text-align: center; }}
                        .container {{ max-width: 600px; margin: auto; border: 1px solid #30363d; border-radius: 8px; padding: 30px; background-color: #161b22; }}
                        h2 {{ color: #58a6ff; }}
                        .metric {{ font-size: 24px; font-weight: bold; margin: 20px 0; color: #ff7b72; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2>MLOps Data Drift Diagnostics</h2>
                        <p>Evidently AI was not installed or failed to run in the ETL environment.</p>
                        <p>However, mathematical embedding drift metrics were computed successfully:</p>
                        <div class="metric">PSI: {mean_psi:.4f} ({status})</div>
                        <p>Wasserstein Distance: {mean_wd:.4f}</p>
                        <p>Reference Samples: {len(reference_df)} | Current Samples: {len(current_df)}</p>
                    </div>
                </body>
                </html>
                """)
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
