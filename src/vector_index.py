import os
import json
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import faiss
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Default local paths
DATA_DIR = os.environ.get("DATA_DIR", "data")
METADATA_FILENAME = "news_metadata.parquet"
INDEX_FILENAME = "index.faiss"

def get_local_paths() -> Tuple[str, str]:
    """
    Returns local paths for metadata and FAISS index, ensuring directories exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    metadata_path = os.path.join(DATA_DIR, METADATA_FILENAME)
    index_path = os.path.join(DATA_DIR, INDEX_FILENAME)
    return metadata_path, index_path

def download_from_hf(repo_id: str, token: Optional[str] = None) -> bool:
    """
    Attempts to download metadata and index files from Hugging Face hub to the local DATA_DIR.
    Returns True if files were successfully downloaded, False otherwise.
    """
    metadata_path, index_path = get_local_paths()
    success = True
    
    # Download Parquet metadata
    try:
        logger.info(f"Attempting to download {METADATA_FILENAME} from Hugging Face dataset: {repo_id}")
        downloaded_metadata = hf_hub_download(
            repo_id=repo_id,
            filename=METADATA_FILENAME,
            repo_type="dataset",
            token=token,
            local_dir=DATA_DIR
        )
        logger.info(f"Downloaded metadata to {downloaded_metadata}")
    except Exception as e:
        logger.warning(f"Could not download {METADATA_FILENAME} from Hugging Face: {e}. Starting fresh or using local copy.")
        success = False

    # Download FAISS index
    try:
        logger.info(f"Attempting to download {INDEX_FILENAME} from Hugging Face dataset: {repo_id}")
        downloaded_index = hf_hub_download(
            repo_id=repo_id,
            filename=INDEX_FILENAME,
            repo_type="dataset",
            token=token,
            local_dir=DATA_DIR
        )
        logger.info(f"Downloaded FAISS index to {downloaded_index}")
    except Exception as e:
        logger.warning(f"Could not download {INDEX_FILENAME} from Hugging Face: {e}. Starting fresh or using local copy.")
        success = False

    return success

def load_index_and_metadata(repo_id: Optional[str] = None, token: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[faiss.Index]]:
    """
    Loads the Parquet metadata and FAISS index.
    Attempts to fetch from Hugging Face first (if repo_id provided), falling back to local files.
    If no data exists, returns an empty DataFrame and None.
    """
    metadata_path, index_path = get_local_paths()
    
    # Try Hugging Face first
    if repo_id:
        download_from_hf(repo_id, token)
        
    # Load metadata
    df = pd.DataFrame()
    if os.path.exists(metadata_path):
        try:
            df = pd.read_parquet(metadata_path)
            logger.info(f"Loaded {len(df)} metadata records from {metadata_path}")
        except Exception as e:
            logger.error(f"Error reading metadata parquet file: {e}")
    else:
        logger.info("No existing metadata parquet found. Will initialize empty.")

    # Load FAISS index
    index = None
    if os.path.exists(index_path) and len(df) > 0:
        try:
            index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors from {index_path}")
            
            # Sanity check
            if index.ntotal != len(df):
                logger.warning(f"Mismatch: FAISS index has {index.ntotal} vectors but Parquet has {len(df)} rows. Rebuilding index...")
                index = rebuild_index_from_df(df)
        except Exception as e:
            logger.error(f"Error reading FAISS index: {e}. Rebuilding index from parquet...")
            index = rebuild_index_from_df(df)
    elif len(df) > 0:
        logger.warning("Parquet metadata exists but FAISS index is missing. Rebuilding index...")
        index = rebuild_index_from_df(df)
        
    return df, index

def rebuild_index_from_df(df: pd.DataFrame) -> Optional[faiss.Index]:
    """
    Rebuilds a FAISS index from the embedding vectors stored in the Parquet DataFrame.
    """
    if df.empty or "embedding" not in df.columns:
        logger.warning("Cannot rebuild index: DataFrame is empty or missing 'embedding' column.")
        return None
        
    try:
        embeddings = np.stack(df["embedding"].values).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Save rebuilt index
        _, index_path = get_local_paths()
        faiss.write_index(index, index_path)
        logger.info(f"Successfully rebuilt and saved FAISS index with {index.ntotal} vectors.")
        return index
    except Exception as e:
        logger.error(f"Failed to rebuild FAISS index: {e}")
        return None

def update_index_and_metadata(
    df: pd.DataFrame, 
    index: Optional[faiss.Index], 
    new_articles: List[Dict]
) -> Tuple[pd.DataFrame, faiss.Index]:
    """
    Appends new articles and their embeddings to the metadata DataFrame and FAISS index.
    Saves the results locally.
    """
    if not new_articles:
        logger.info("No new articles to add.")
        if index is None:
            # Create a dummy index with 384 dimensions if no articles exist yet
            index = faiss.IndexFlatIP(384)
        return df, index

    metadata_path, index_path = get_local_paths()
    
    # Prepare new article rows for DataFrame
    new_rows = []
    new_embeddings = []
    
    for art in new_articles:
        emb = art["embedding"]
        new_embeddings.append(emb)
        
        # Strip embedding from dictionary to avoid redundant columns if needed,
        # but we do want to store it as a fallback in Parquet.
        row = {
            "article_id": art["article_id"],
            "timestamp": art["timestamp"],
            "source": art["source"],
            "url": art["url"],
            "title_de": art["title_de"],
            "body_de": art["body_de"],
            "entities": json.dumps(art["entities"]), # Store as JSON string for safety
            "embedding": list(emb.astype(float)) # Convert numpy array to list for parquet compatibility
        }
        new_rows.append(row)
        
    new_df = pd.DataFrame(new_rows)
    new_embeddings_np = np.stack(new_embeddings).astype('float32')
    
    # 1. Update FAISS Index
    if index is None:
        dimension = new_embeddings_np.shape[1]
        logger.info(f"Initializing new FAISS IndexFlatIP with dimension {dimension}")
        index = faiss.IndexFlatIP(dimension)
        
    index.add(new_embeddings_np)
    
    # 2. Update Parquet DataFrame
    if df.empty:
        updated_df = new_df
    else:
        updated_df = pd.concat([df, new_df], ignore_index=True)
        
    # Save files locally
    try:
        updated_df.to_parquet(metadata_path, index=False)
        faiss.write_index(index, index_path)
        logger.info(f"Successfully saved updated metadata ({len(updated_df)} rows) and FAISS index ({index.ntotal} vectors) locally.")
    except Exception as e:
        logger.error(f"Error saving updated index or metadata: {e}")
        raise e
        
    return updated_df, index

def query_vector_search(
    query_vector: np.ndarray, 
    df: pd.DataFrame, 
    index: faiss.Index, 
    top_k: int = 5
) -> List[Dict]:
    """
    Searches the FAISS index for the closest vector match and maps back to the Parquet metadata.
    """
    if df.empty or index is None or index.ntotal == 0:
        logger.warning("Empty search database or FAISS index.")
        return []
        
    # Ensure query vector is float32 and shape is (1, dimension)
    q_vec = query_vector.astype('float32')
    if len(q_vec.shape) == 1:
        q_vec = np.expand_dims(q_vec, axis=0)
        
    # Search the index
    # distances represents dot product/cosine similarity score because we normalized vectors
    similarities, indices = index.search(q_vec, min(top_k, index.ntotal))
    
    results = []
    for score, idx in zip(similarities[0], indices[0]):
        if idx < 0 or idx >= len(df):
            continue
            
        row = df.iloc[idx]
        
        # Load and parse entities safely
        try:
            entities = json.loads(row["entities"])
        except Exception:
            entities = []
            
        results.append({
            "article_id": row["article_id"],
            "timestamp": row["timestamp"],
            "source": row["source"],
            "url": row["url"],
            "title_de": row["title_de"],
            "body_de": row["body_de"],
            "entities": entities,
            "similarity_score": float(score)
        })
        
    return results

if __name__ == "__main__":
    # Test index creation and querying
    logging.basicConfig(level=logging.INFO)
    
    # Setup dummy data
    dummy_articles = [
        {
            "article_id": "test_1",
            "timestamp": "2026-06-14T12:00:00",
            "source": "Tagesschau",
            "url": "https://example.com/test_1",
            "title_de": "Dummy Titel",
            "body_de": "Dies ist ein Testartikel über künstliche Intelligenz in Hamburg.",
            "entities": [{"word": "Hamburg", "entity": "LOC", "score": 0.99}],
            "embedding": np.random.randn(384).astype('float32')
        }
    ]
    # L2 normalize the dummy embedding
    dummy_articles[0]["embedding"] = dummy_articles[0]["embedding"] / np.linalg.norm(dummy_articles[0]["embedding"])
    
    # Test clean build
    # Force clean data directory for testing
    if os.path.exists("test_data"):
        import shutil
        shutil.rmtree("test_data")
    os.environ["DATA_DIR"] = "test_data"
    
    df, index = load_index_and_metadata()
    print(f"Initial DB: {len(df)} rows, Index: {index}")
    
    df, index = update_index_and_metadata(df, index, dummy_articles)
    print(f"Updated DB: {len(df)} rows, Index size: {index.ntotal}")
    
    # Test search query
    q = np.random.randn(384).astype('float32')
    q = q / np.linalg.norm(q)
    res = query_vector_search(q, df, index, top_k=1)
    print(f"Search result score: {res[0]['similarity_score']}, title: {res[0]['title_de']}")
    
    # Cleanup test dir
    import shutil
    shutil.rmtree("test_data")
