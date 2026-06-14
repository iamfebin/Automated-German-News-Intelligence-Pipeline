import os
import logging
from typing import Optional
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

# Default configurations from environment variables
HF_REPO_ID = os.environ.get("HF_REPO_ID")
HF_WRITE_TOKEN = os.environ.get("HF_WRITE_TOKEN")
DATA_DIR = os.environ.get("DATA_DIR", "data")

def push_to_hub(repo_id: Optional[str] = None, token: Optional[str] = None, folder_path: str = DATA_DIR) -> bool:
    """
    Pushes the contents of folder_path (Parquet, FAISS, HTML drift reports)
    to the target Hugging Face dataset repository.
    Creates the repository if it doesn't exist.
    """
    # Fallback to env variables if parameters not passed
    repo_id = repo_id or HF_REPO_ID
    token = token or HF_WRITE_TOKEN
    
    if not repo_id or not token:
        logger.warning(
            "Hugging Face credentials missing (HF_REPO_ID or HF_WRITE_TOKEN not set). "
            "Skipping Hugging Face Hub upload. Running in Local-Only Mode."
        )
        return False
        
    try:
        logger.info(f"Pusing local artifacts from {folder_path} to HF Hub Dataset: {repo_id}...")
        api = HfApi()
        
        # Create dataset repo if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
            token=token
        )
        
        # Upload folder contents
        future = api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Update German News Intelligence Index & drift diagnostics"
        )
        
        logger.info(f"Successfully uploaded data folder to HF Hub: {future}")
        return True
    except Exception as e:
        logger.error(f"Failed to push to Hugging Face Hub: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Check execution
    push_to_hub()
