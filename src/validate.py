import os
import sys
import logging
import unittest
import numpy as np
import pandas as pd
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import clean_german_text, generate_article_id, RSS_FEEDS
from src.drift_monitor import calculate_psi, compute_embedding_drift
from src.vector_index import query_vector_search

class TestPipelineComponents(unittest.TestCase):

    def test_clean_german_text(self):
        logger.info("Running test_clean_german_text...")
        self.assertEqual(clean_german_text("<p>Hallo Welt!</p>"), "Hallo Welt!")
        self.assertEqual(clean_german_text("M&uuml;nchen"), "München")
        self.assertEqual(clean_german_text("Gro&szlig;e Stra&szlig;e"), "Große Straße")
        # Unicode NFC normalization check
        input_str = "a\u0308" # 'a' + combining diaeresis
        cleaned = clean_german_text(input_str)
        self.assertEqual(cleaned, "ä")

    def test_article_id_generation(self):
        logger.info("Running test_article_id_generation...")
        url = "https://www.tagesschau.de/inland/test-article-100.html"
        pub_date = "2026-06-14T12:00:00"
        
        id1 = generate_article_id(url, pub_date)
        id2 = generate_article_id(url, pub_date)
        id3 = generate_article_id(url, "2026-06-14T12:00:01")
        
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertEqual(len(id1), 64) # SHA-256 length

    def test_calculate_psi(self):
        logger.info("Running test_calculate_psi...")
        np.random.seed(42)
        expected = np.random.normal(0, 1, 100)
        # Shifted distribution
        actual = np.random.normal(0.5, 1, 100)
        
        psi_val = calculate_psi(expected, actual)
        logger.info(f"Calculated PSI: {psi_val:.4f}")
        self.assertTrue(psi_val > 0.0)

    def test_embedding_drift(self):
        logger.info("Running test_embedding_drift...")
        np.random.seed(42)
        ref = np.random.normal(0, 1, (20, 10))
        cur = np.random.normal(0.2, 1, (10, 10))
        
        wd, psi = compute_embedding_drift(ref, cur)
        logger.info(f"Embedding Drift - WD: {wd:.4f}, PSI: {psi:.4f}")
        self.assertTrue(wd > 0)
        self.assertTrue(psi > 0)

    def test_vector_search_alignment(self):
        logger.info("Running test_vector_search_alignment...")
        # Create a simple flat IP index
        dim = 4
        index = faiss.IndexFlatIP(dim)
        
        # Vectors
        vecs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype='float32')
        index.add(vecs)
        
        # Metadata
        df = pd.DataFrame([
            {"article_id": "1", "title_de": "A", "entities": "[]", "source": "Tagesschau", "url": "U1", "body_de": "B1", "timestamp": "T1"},
            {"article_id": "2", "title_de": "B", "entities": "[]", "source": "Spiegel", "url": "U2", "body_de": "B2", "timestamp": "T2"},
            {"article_id": "3", "title_de": "C", "entities": "[]", "source": "DW", "url": "U3", "body_de": "B3", "timestamp": "T3"}
        ])
        
        # Query closest to vector 2
        query = np.array([0.1, 0.9, 0.0, 0.0], dtype='float32')
        query = query / np.linalg.norm(query)
        
        results = query_vector_search(query, df, index, top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["article_id"], "2")
        self.assertEqual(results[0]["title_de"], "B")
        self.assertTrue(results[0]["similarity_score"] > results[1]["similarity_score"])

if __name__ == "__main__":
    unittest.main()
