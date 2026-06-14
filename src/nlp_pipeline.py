import os
import json
import logging
from typing import List, Dict, Any
import numpy as np
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Choose models based on environment variables or default to lightweight models
NER_MODEL_NAME = os.environ.get("NER_MODEL_NAME", "fhswf/bert_de_ner")
TRANSLATION_MODEL_NAME = os.environ.get("TRANSLATION_MODEL_NAME", "Helsinki-NLP/opus-mt-de-en")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

class NLPPipeline:
    def __init__(self):
        self.ner_pipeline = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.embedding_model = None
        
        # Check if CUDA is available, otherwise default to CPU
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")

    def load_ner(self):
        if self.ner_pipeline is None:
            logger.info(f"Loading NER model: {NER_MODEL_NAME}...")
            self.ner_pipeline = pipeline(
                "ner", 
                model=NER_MODEL_NAME, 
                aggregation_strategy="simple",
                device=self.device
            )
        return self.ner_pipeline

    def load_translation(self):
        if self.translation_model is None or self.translation_tokenizer is None:
            logger.info(f"Loading Translation model: {TRANSLATION_MODEL_NAME}...")
            from transformers import MarianMTModel, MarianTokenizer
            self.translation_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
            self.translation_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME)
            if self.device == 0:
                self.translation_model = self.translation_model.to("cuda")
        return self.translation_model, self.translation_tokenizer

    def load_embeddings(self):
        if self.embedding_model is None:
            logger.info(f"Loading Embedding model: {EMBEDDING_MODEL_NAME}...")
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device_str)
        return self.embedding_model

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs Named Entity Recognition on the German text.
        Returns a list of entities with words, types, and confidence scores.
        """
        if not text:
            return []
        
        try:
            ner = self.load_ner()
            # Truncate text to fit model context window (typically 512 tokens)
            # A rough estimate is 1500 characters.
            truncated_text = text[:1500]
            results = ner(truncated_text)
            
            entities = []
            for item in results:
                entities.append({
                    "word": str(item.get("word", "")),
                    "entity": str(item.get("entity_group", "")),
                    "score": float(item.get("score", 0.0))
                })
            return entities
        except Exception as e:
            logger.error(f"Error during NER extraction: {e}")
            return []

    def generate_summary_en(self, text: str) -> str:
        """
        Generates an English summary by translating the lead paragraphs/sentences of the article.
        """
        if not text:
            return ""
        
        try:
            # Extract lead content (e.g. first 3 sentences or up to ~600 chars)
            # Find the first few sentences
            sentences = text.split(". ")
            lead_text = ". ".join(sentences[:3])
            if len(lead_text) > 600:
                lead_text = lead_text[:600]
            if not lead_text.endswith("."):
                lead_text += "."
                
            model, tokenizer = self.load_translation()
            
            # Tokenize and generate translation
            inputs = tokenizer(lead_text, return_tensors="pt", padding=True, truncation=True)
            if self.device == 0:
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
            translated = model.generate(**inputs)
            summary = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return summary.strip()
        except Exception as e:
            logger.error(f"Error during translation/summary generation: {e}")
            return "Translation unavailable."

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates multilingual sentence embeddings for a list of texts.
        Embeddings are normalized to unit L2 length to support cosine similarity.
        """
        if not texts:
            return np.empty((0, 0))
            
        try:
            model = self.load_embeddings()
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
            # Normalize embeddings to L2 unit vectors for Cosine similarity search
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            normalized_embeddings = embeddings / norms
            
            return normalized_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise e

if __name__ == "__main__":
    # Quick pipeline self-test
    logging.basicConfig(level=logging.INFO)
    pipeline = NLPPipeline()
    
    test_german_text = "Angela Merkel besuchte gestern die Siemens AG in München."
    print("Testing German NER...")
    entities = pipeline.extract_entities(test_german_text)
    print(f"Entities: {json.dumps(entities, indent=2)}")
    
    print("\nTesting Translation (Summary)...")
    summary = pipeline.generate_summary_en(test_german_text)
    print(f"English translation: {summary}")
    
    print("\nTesting Embeddings...")
    emb = pipeline.generate_embeddings([test_german_text])
    print(f"Embedding shape: {emb.shape}")
    print(f"L2 Norm (should be 1.0): {np.linalg.norm(emb[0])}")
