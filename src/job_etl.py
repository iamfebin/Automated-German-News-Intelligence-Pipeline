import os
import sys
import logging
import warnings
from datetime import datetime
from typing import List, Dict

# Suppress SyntaxWarnings from third-party libraries (like evidently under Python 3.12+)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add parent directory to path so imports work when executed from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import scrape_news_feeds
from src.nlp_pipeline import NLPPipeline
from src.vector_index import load_index_and_metadata, update_index_and_metadata
from src.drift_monitor import generate_drift_report
from src.push_to_hub import push_to_hub

def main():
    logger.info("=========================================================")
    logger.info("Starting Multilingual German News Intelligence ETL Pipeline")
    logger.info(f"Execution time: {datetime.utcnow().isoformat()} UTC")
    logger.info("=========================================================")
    
    # Retrieve environment settings
    hf_repo_id = os.environ.get("HF_REPO_ID")
    # Support both write token names
    hf_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    limit_per_feed = int(os.environ.get("SCRAPER_LIMIT_PER_FEED", "10"))
    
    if hf_repo_id:
        logger.info(f"Target Hugging Face Dataset: {hf_repo_id}")
    else:
        logger.info("No HF_REPO_ID configured. Pipeline will run in local-only mode.")
        
    # Step 1: Load existing dataset and vector index
    logger.info("Step 1: Loading existing dataset metadata and vector index...")
    df, index = load_index_and_metadata(repo_id=hf_repo_id, token=hf_token)
    
    existing_ids = set()
    if not df.empty and "article_id" in df.columns:
        existing_ids = set(df["article_id"].tolist())
    logger.info(f"Loaded database contains {len(existing_ids)} articles.")
    
    # Step 2: Scrape news feeds
    logger.info(f"Step 2: Scraping latest German news articles (limit={limit_per_feed} per feed)...")
    scraped_articles = scrape_news_feeds(limit_per_feed=limit_per_feed)
    logger.info(f"Scraped total of {len(scraped_articles)} articles.")
    
    # Step 3: Deduplicate articles
    logger.info("Step 3: Deduplicating articles...")
    new_articles: List[Dict] = []
    for art in scraped_articles:
        if art["article_id"] not in existing_ids:
            new_articles.append(art)
            
    logger.info(f"Found {len(new_articles)} new articles to process after deduplication.")
    
    if not new_articles:
        logger.info("No new articles to process. Index is up to date.")
        # Run drift analysis with empty list to update status/timestamps if needed
        logger.info("Updating drift reports for alignment...")
        generate_drift_report([])
        
        # Sync to hub anyway to ensure everything is aligned
        if hf_repo_id:
            push_to_hub(repo_id=hf_repo_id, token=hf_token)
            
        logger.info("ETL Pipeline complete. Exiting successfully.")
        return
        
    # Step 4: Run NLP Pipeline on new articles
    logger.info("Step 4: Loading NLP Models and extracting features (NER, Translation, Embeddings)...")
    nlp = NLPPipeline()
    
    processed_count = 0
    valid_new_articles = []
    for i, art in enumerate(new_articles):
        logger.info(f"[{i+1}/{len(new_articles)}] Processing: {art['title_de']}")
        
        try:
            # 4.1 Localized Named Entity Recognition
            art["entities"] = nlp.extract_entities(art["body_de"])
            
            # 4.2 Embeddings vector generation (Joint Multilingual Vector Space)
            # Embed title and body concatenated
            text_to_embed = f"{art['title_de']}. {art['body_de']}"
            embeddings = nlp.generate_embeddings([text_to_embed])
            art["embedding"] = embeddings[0]
            
            valid_new_articles.append(art)
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process article '{art['title_de']}': {e}")
            
    logger.info(f"Successfully processed {processed_count} of {len(new_articles)} new articles.")
    
    # Step 5: Update Vector Index and Parquet Metadata
    logger.info("Step 5: Updating FAISS vector index and Parquet metadata store...")
    df, index = update_index_and_metadata(df, index, valid_new_articles)
    
    # Step 6: Execute Data Drift Diagnostics
    logger.info("Step 6: Executing MLOps Data Drift diagnostics...")
    new_article_ids = [art["article_id"] for art in valid_new_articles]
    drift_metrics = generate_drift_report(new_article_ids)
    
    if drift_metrics:
        logger.info(f"Drift Analysis completed. Embedding PSI: {drift_metrics['embedding_drift']['population_stability_index']:.4f} Status: {drift_metrics['embedding_drift']['status']}")
    
    # Step 7: Push updated artifacts to Hugging Face Hub
    if hf_repo_id:
        logger.info("Step 7: Pushing updated database, index, and reports to Hugging Face...")
        push_to_hub(repo_id=hf_repo_id, token=hf_token)
    else:
        logger.info("Step 7: Skipping Hugging Face upload (local mode). Files are available in local 'data/' directory.")
        
    logger.info("=========================================================")
    logger.info("ETL Job Execution Complete. All steps ran successfully!")
    logger.info("=========================================================")

if __name__ == "__main__":
    main()
