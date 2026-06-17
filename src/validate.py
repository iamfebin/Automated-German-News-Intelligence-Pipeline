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

from src.scraper import clean_german_text, generate_article_id, RSS_FEEDS, is_paywalled_content
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

    def test_is_paywalled_content(self):
        logger.info("Running test_is_paywalled_content...")
        
        # Test Case 1: Paywalled Spiegel+ URL
        self.assertTrue(is_paywalled_content(
            title="Normal Title",
            url="https://www.spiegel.de/plus/some-premium-article-12345.html",
            body="Some normal German text content."
        ))
        
        # Test Case 2: Title indicating Spiegel+
        self.assertTrue(is_paywalled_content(
            title="SPIEGEL+: Premium News",
            url="https://www.spiegel.de/panorama/some-article-12345.html",
            body="Some normal German text content."
        ))
        
        # Test Case 3: Body containing the paywall message reported by user
        paywall_body = (
            "Sie können den Artikel nicht mehr aufrufen. Der Link, der Ihnen zugesendet wurde, "
            "ist entweder älter als 30 Tage oder der Artikel wurde bereits 10 Mal geöffnet. "
            "Haben Sie bereits ein Digital-Abonnement? Zum Login."
        )
        self.assertTrue(is_paywalled_content(
            title="Random Article",
            url="https://www.spiegel.de/panorama/normal-url.html",
            body=paywall_body
        ))
        
        # Test Case 4: Body containing typical subscription boilerplate
        self.assertTrue(is_paywalled_content(
            title="Another News",
            url="https://www.spiegel.de/panorama/normal-url.html",
            body="Dieser Text ist exklusiv für Abonnenten. Bitte abonnieren Sie Spiegel+."
        ))
        
        # Test Case 5: Normal non-paywalled article
        self.assertFalse(is_paywalled_content(
            title="Normal Article Title",
            url="https://www.tagesschau.de/inland/normal-article.html",
            body="Das ist ein ganz normaler Nachrichtenartikel ohne Bezahlschranke."
        ))

if __name__ == "__main__":
    unittest.main()
