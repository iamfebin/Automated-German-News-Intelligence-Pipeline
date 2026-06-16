# Proposed Architectural Changes: Multilingual Vector Space Realignment

## 1. Objective

To correct the search inaccuracy caused by early translation and monolingual embedding, we will refactor the platform to use a **Joint Multilingual Vector Space**. This ensures both English and German queries map natively to German documents without translation distortion, while strictly adhering to the project's zero-cost, memory-constrained (GitHub Actions / Streamlit) limitations.

## 2. Refactoring Plan by Component

### A. The ETL & Embedding Pipeline (`src/job_etl.py` & `src/nlp_pipeline.py`)

**Current State:** Translates German to English -> Embeds English text using `all-MiniLM-L6-v2`. **Proposed State:** Embeds raw German text directly using a lightweight multilingual model.

-   **Action 1: Swap the Embedding Model.** Replace `sentence-transformers/all-MiniLM-L6-v2` with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
    
    -   _Benefit:_ At ~470MB, it fits easily within the GHA 7GB RAM limit while natively supporting 50+ languages in the same semantic space.
        
-   **Action 2: Remove Batch Translation.** Delete the `Helsinki-NLP/opus-mt-de-en` pipeline from the ETL process entirely.
    
    -   _Benefit:_ Massively reduces GitHub Actions compute time and prevents translation errors from permanently poisoning the vector index.
        
-   **Action 3: Update Target Payload.** Pass the cleaned `body_de` (or a concatenation of `title_de` + `body_de`) directly into the multilingual embedding model.
    

### B. The Database Schema (`news_metadata.parquet`)

**Current State:** Includes a pre-computed `summary_en` column. **Proposed State:** Removes pre-computed translations to save storage and enforce query-time translation.

-   **Action 1: Drop the `summary_en` column** from the daily batch generation. The Parquet file will now strictly serve as a lightweight metadata and vector storage mechanism: `[article_id, timestamp, source, url, title_de, body_de, entities, embedding]`.
    

### C. The Interactive UI & Search Engine (`app.py`)

**Current State:** Matches English queries to English document vectors. **Proposed State:** Matches multilingual queries to German document vectors, translating only the final output.

-   **Action 1: Multilingual Query Vectorization.** Update the Hugging Face Serverless API call (or local fallback) to use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` when vectorizing the user's search query.
    
-   **Action 2: Just-In-Time (JIT) Translation.** Once FAISS retrieves the Top-$K$ relevant Parquet records, initialize the translation API/model _only_ for those 3 to 5 articles.
    
    -   _Execution:_ Send the retrieved `body_de` to the HF Serverless API for `Helsinki-NLP/opus-mt-de-en` to display an English summary to the user.
        
    -   _Benefit:_ Preserves Streamlit memory and drastically cuts down translation overhead.
        

### D. Text Normalization (`src/scraper.py`)

**Current State:** Basic HTML stripping and Unicode normalization. **Proposed State:** Enhanced preparation for the multilingual tokenizer.

-   **Action 1: Compound Word Safety.** Ensure hyphens and punctuation around typical German compound words are stripped cleanly. Because we are shifting to `paraphrase-multilingual-MiniLM-L12-v2`, the model's native SentencePiece tokenizer is highly optimized for German sub-word chunks (e.g., breaking `Bundestagswahl` into sub-tokens). We do not need a heavy external library, provided the text is cleanly formatted.
    

## 3. Expected Outcomes

1.  **Search Accuracy:** Drastically improved. German queries will work natively, and English queries will map accurately to German semantic concepts.
    
2.  **Resource Efficiency:** GitHub Actions execution time will drop significantly by removing the heavy translation loop.
    
3.  **Data Integrity:** The FAISS index will reflect the true intent of the original journalists, uncontaminated by machine translation hallucinations.
