# Software Requirements Specification (SRS) & System Design Document

## Project: Multilingual German News Intelligence Platform (Zero-Cost Architecture)

## Part 1: Software Requirements Specification (SRS)

### 1. Introduction

#### 1.1 Purpose

This document specifies the software requirements and architectural design for the **Multilingual German News Intelligence Platform**. The platform is a production-grade, zero-operational-cost NLP utility designed to ingest, process, embed, index, and monitor German-language news media. It enables English-speaking users to query regional German media semantically in English, returning contextual multilingual summaries and localized Named Entity Recognition (NER), alongside an automated MLOps data drift dashboard.

#### 1.2 Scope

The scope of this project covers:

1.  **Serverless Data Ingestion**: Daily cron-scheduled scraping of top German news outlets (e.g., Tagesschau, Deutsche Welle, Der Spiegel) via RSS and HTML parsing.
    
2.  **Resource-Constrained Feature Engineering**: Text cleaning, compound-word parsing, and localized German NER.
    
3.  **Cross-Lingual Representation**: Pre-computing vector representations using a multilingual sentence transformer.
    
4.  **Zero-Cost Versioned Storage**: Utilizing Hugging Face Datasets as an immutable, free object store for metadata (`.parquet`) and vector indices (`faiss_index.bin`).
    
5.  **Inference & Search Portal**: A Streamlit interface that pulls the latest indices, embeds natural language English queries, performs vector matching, and serves German/English summaries.
    
6.  **LLMOps and Data Drift Monitoring**: Evaluating incoming text and embedding shifts daily using statistical metrics (e.g., Wasserstein Distance) visualized via Evidently AI.
    

#### 1.3 Key Definitions & Acronyms

-   **NER**: Named Entity Recognition
    
-   **FAISS**: Facebook AI Similarity Search
    
-   **HF**: Hugging Face
    
-   **GHA**: GitHub Actions
    
-   **PSI**: Population Stability Index
    
-   **WD**: Wasserstein Distance
    

### 2. Overall Description

#### 2.1 Product Perspective

Unlike typical MLOps architectures requiring expensive cloud databases (Pinecone, AWS RDS) and persistent virtual machines (EC2, Kubernetes), this platform operates entirely within a **decoupled, serverless, zero-budget ecosystem**. By shifting the computationally heavy embedding and indexing pipelines to free CI/CD runners (GitHub Actions) and using lightweight client-side loading (Streamlit + FAISS), we eliminate runtime cloud costs while keeping the data index fresh.

```
+-----------------------------------+
|     GitHub Actions (Scheduler)    |  <-- Run daily (Compute Embeddings, NER, Drift)
+-----------------------------------+
                  |
                  v  (Write Parquet + FAISS Index)
+-----------------------------------+
|       Hugging Face Datasets       |  <-- Free, Version-Controlled Storage
+-----------------------------------+
                  |
                  v  (On-Demand Stream Loading)
+-----------------------------------+
|     Streamlit Cloud (Hosting)     |  <-- Free Public-Facing UI Web Portal
+-----------------------------------+

```

#### 2.2 Design and Implementation Constraints

1.  **$0.00 Budget**: Absolute restriction against any paid cloud APIs, databases, or hosting providers. No registration steps requiring a credit card are allowed.
    
2.  **Ephemeral Compute Limits**: GitHub Actions workflows are limited to 6 hours of execution time per run (which is highly generous for daily incremental scrapes) and 7 GB of RAM.
    
3.  **Streamlit RAM Limit**: Streamlit Community Cloud instances are capped at 1 GB of RAM. The production application must never cache excessive data or load overly large transformer models directly into the web application memory.
    

### 3. Functional Requirements

#### 3.1 Scraper & ETL Ingestion Pipeline

-   **FR-1.1**: The system shall run automatically once per day at a configured cron time (e.g., `0 6 * * *`) via GitHub Actions.
    
-   **FR-1.2**: The scraper shall parse RSS feeds from at least three pre-defined German news portals.
    
-   **FR-1.3**: The system shall deduplicate incoming articles against existing database records using a unique cryptographic hash (SHA-256) of the article URL and publication date.
    
-   **FR-1.4**: The text extraction phase must clean HTML tags, normalize German Umlauts (`ä`, `ö`, `ü`), and format the standard German `ß` cleanly.
    

#### 3.2 Transformer Inference & Vector Indexing Pipeline

-   **FR-2.1**: The system shall extract Named Entities (locations, companies, political figures) using a lightweight German-specific model (e.g., `dbmdz/bert-large-cased-finetuned-conll03-germeval`).
    
-   **FR-2.2**: The ingestion runner shall generate sentence embeddings for each news article using a multilingual encoder (e.g., `sentence-transformers/LaBSE` or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
    
-   **FR-2.3**: The system shall build an incremental FAISS vector index using L2 Euclidean distance or Inner Product (Cosine similarity) metrics.
    
-   **FR-2.4**: The runner must commit the updated metadata file (`news_metadata.parquet`) and the serialized FAISS index file (`index.faiss`) directly to a target Hugging Face Dataset repository.
    

#### 3.3 Semantic Search & Summarization (UI)

-   **FR-3.1**: The web UI shall offer a search input field accepting queries in English or German.
    
-   **FR-3.2**: When a search is initiated, the app shall vectorize the query using the Hugging Face Serverless Inference API (to preserve the web container's 1GB RAM budget).
    
-   **FR-3.3**: The UI shall perform a vector similarity search across the loaded FAISS index and return the top-$K$ most contextually relevant German articles.
    
-   **FR-3.4**: For each retrieved article, the UI shall display the original German text, localized metadata, extracted entities, and a bilingual translation/summary.
    

#### 3.4 Data Drift & MLOps Analytics

-   **FR-4.1**: The daily pipeline shall compare the statistical properties of the newly scraped text embeddings with a rolling baseline of the past 14 days.
    
-   **FR-4.2**: The pipeline shall output an Evidently AI data drift report in HTML format and log statistical metrics.
    
-   **FR-4.3**: The Streamlit interface must render these drift reports inside a dedicated "MLOps Health" tab, enabling recruiters to inspect data distribution shifts in real-time.
    

## Part 2: System Design Document (SDD)

### 1. Architectural Design & Data Flow

This application is split into two asynchronous modules: the **Data & Index Compilation Engine (Batch)** and the **Interactive Search Engine (Real-Time Web Portal)**.

```
       +--------------------------------------------------------------------------+
       |                  BATCH ENGINE (GitHub Actions Runner)                    |
       +--------------------------------------------------------------------------+
       |                                                                          |
       |  [RSS Feeds] -> [Scraper] -> [Clean Text]                                |
       |                                     |                                    |
       |  [Local Embeddings Model] <---------+                                    |
       |             |                                                            |
       |             v                                                            |
       |  [Generate Embeddings & NER]                                             |
       |             |                                                            |
       |             v                                                            |
       |  [Update Parquet DataFrame + FAISS Index]                                |
       |             |                                                            |
       |             v                                                            |
       |  [Evidently AI Drift Check]                                              |
       |             |                                                            |
       |             +------> Commit Files to -> [Hugging Face Dataset Repo]      |
       +--------------------------------------------------------------------------+
                                                               |
                                                               v
       +--------------------------------------------------------------------------+
       |                     REAL-TIME PORTAL (Streamlit Cloud)                   |
       +--------------------------------------------------------------------------+
       |                                                                          |
       |  [User Enters English Query]                                             |
       |             |                                                            |
       |             v                                                            |
       |  [HF Serverless API] ----> Embed Query                                   |
       |                                |                                         |
       |                                v                                         |
       |  [Search local FAISS index] <---+                                         |
       |             |                                                            |
       |             v                                                            |
       |  [Retrieve Top-K Parquet Records]                                        |
       |             |                                                            |
       |             v                                                            |
       |  [Render Bilingual Summaries, NER Tags, and Drift Diagnostics]            |
       +--------------------------------------------------------------------------+

```

### 2. Component Specification

#### 2.1 Component A: ETL & Embedding Generator (`job_etl.py`)

This component executes inside GitHub Actions.

-   **Linguistic Prep**: German compounds can artificially blow up vocabularies. The ETL pipeline will normalize input strings and extract semantic elements.
    
-   **Entity Extraction**:
    
    ```
    from transformers import pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-germeval", device=-1)
    
    ```
    
-   **Multilingual Embedding Vectorization**: We use `sentence-transformers/LaBSE` because it maps 109 languages into a joint embedding space. The similarity score between an English query vector $Q$ and a German document vector $D$ is computed as:
    
    $$CosineSimilarity(Q, D) = \frac{Q \cdot D}{\|Q\| \|D\|}$$

#### 2.2 Component B: Storage & Versioning (Hugging Face Hub API)

Rather than managing AWS S3 access keys or configuring complex DB connections, data storage is abstracted into Git-based LFS via the Hugging Face Hub.

-   **Dataset Repo**: `datasets/username/german-news-intelligence`
    
-   **Artifacts**:
    
    -   `news_metadata.parquet`: Compressed columnar data containing `[id, url, title, raw_german_text, publication_date, entities, summary_en]`.
        
    -   `index.faiss`: Binary indexing structure.
        
    -   `drift_report.html`: The generated Evidently AI interactive monitoring page.
        

#### 2.3 Component C: Vector Search & Presentation (`app.py`)

This component runs inside Streamlit.

-   **Initialization**: On boot, the app fetches `news_metadata.parquet` and `index.faiss` from Hugging Face using standard HTTP requests:
    
    ```
    import pandas as pd
    import faiss
    import requests
    
    # Load metadata
    df = pd.read_parquet("[https://huggingface.co/datasets/username/german-news-intelligence/resolve/main/news_metadata.parquet](https://huggingface.co/datasets/username/german-news-intelligence/resolve/main/news_metadata.parquet)")
    
    # Load FAISS
    r = requests.get("[https://huggingface.co/datasets/username/german-news-intelligence/resolve/main/index.faiss](https://huggingface.co/datasets/username/german-news-intelligence/resolve/main/index.faiss)")
    with open("index.faiss", "wb") as f:
        f.write(r.content)
    index = faiss.read_index("index.faiss")
    
    ```
    
-   **Inference Guardrail**: To avoid memory limits (OOM errors) in Streamlit, query vectorization is offloaded via an API request to Hugging Face's serverless endpoints:
    
    ```
    API_URL = "[https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/LaBSE](https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/LaBSE)"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    def query_vector(text):
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        return response.json()
    
    ```
    

#### 2.4 Component D: MLOps Drift Engine (`drift_monitor.py`)

As global news topics fluctuate, the linguistic profile of crawled text shifts. We monitor this concept drift to show recruiters you understand model performance degradation in dynamic environments.

-   **Statistical Methods**:
    
    -   **Wasserstein Distance (Earth Mover's Distance)**: Measures the minimum work needed to transform the embedding distribution of incoming validation samples $P(x)$ to match the baseline distribution $Q(x)$:
        
        $$W(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$
    -   **Population Stability Index (PSI)**: Quantifies how much a variable has shifted over time:
        
        $$PSI = \sum_{i=1}^{B} \left( (Actual_i - Expected_i) \times \ln\left(\frac{Actual_i}{Expected_i}\right) \right)$$
        
        Where $B$ is the number of bins, $Actual_i$ is the distribution of the active dataset, and $Expected_i$ is the baseline dataset.
        
-   **Implementation**: We utilize **Evidently AI**'s standard Text Drift presets to calculate shifts in text length, vocabulary diversity, and semantic embedding vectors, producing a comprehensive web report automatically saved to our Hugging Face directory.
    

### 3. Database Schema & Data Dictionary

#### Column Definitions (`news_metadata.parquet`)

Column Name

Data Type

Description

`article_id`

`VARCHAR(64)`

SHA-256 unique identifier generated from URL

`timestamp`

`TIMESTAMP`

Extraction datetime

`source`

`VARCHAR(32)`

Name of the publisher (e.g., "Tagesschau")

`url`

`VARCHAR(512)`

Direct link to original German article

`title_de`

`TEXT`

Original title in German

`body_de`

`TEXT`

Raw extracted body text of the article

`entities`

`JSON`

List of dictionary objects: `[{'word': 'Scholz', 'entity': 'PER', 'score': 0.99}]`

`summary_en`

`TEXT`

Translated summary generated by local BART/Mistral or serverless model

### 4. Implementation & CI/CD Pipeline Configuration

To deploy and maintain this infrastructure at **zero cost**, we utilize a GitHub repository with two workflows.

#### 4.1 GitHub Actions Workflow: Ingestion & Feature Generation (`.github/workflows/daily_pipeline.yml`)

```
name: Daily German News Intelligence Pipeline

on:
  schedule:
    - cron: '0 6 * * *' # Every day at 6:00 AM UTC
  workflow_dispatch: # Allows manual trigger for testing

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Codebase
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-node-version: '3.10'
          cache: 'pip'

      - name: Install Dependencies
        run: |
          pip install -r requirements_etl.txt

      - name: Run Scraper & Model Embedder
        env:
          HF_WRITE_TOKEN: ${{ secrets.HF_WRITE_TOKEN }}
        run: |
          python src/job_etl.py

      - name: Execute Drift Diagnostics
        run: |
          python src/drift_monitor.py

      - name: Push Artifacts to Hugging Face
        env:
          HF_WRITE_TOKEN: ${{ secrets.HF_WRITE_TOKEN }}
        run: |
          python src/push_to_hub.py

```

### 5. Architectural Trade-offs & Engineering Decisions

#### 5.1 Pre-computing Embeddings vs. Live API Embeddings

-   **Trade-off**: Generating embeddings on-the-fly for thousands of articles in a free Streamlit instance is impossible due to memory limitations.
    
-   **Decision**: All German news text is embedded **batch-style** during the GitHub Actions workflow runner. The Streamlit client only embeds the single user query and handles the vector math locally using FAISS. This architecture scales beautifully because the CPU overhead of searching a serialized index is negligible compared to active batch embedding generation.
    

#### 5.2 Decoupled FAISS Index vs. Managed Vector Database

-   **Trade-off**: Managed vector databases (e.g., Pinecone, Milvus) have limited free-tier capacity, restrict usage time, or require credit card details.
    
-   **Decision**: We store the raw vector index in a highly compressed static FAISS binary format (`.faiss`). Since the total number of ingested articles over a year is under $10^5$, the file size will remain under 100MB. Downloading and loading a 100MB index on application startup takes less than 3 seconds on Streamlit's free container and avoids all database hosting fees.
    

#### 5.3 Local Multilingual Model vs. Translation APIs

-   **Trade-off**: Translating articles via commercial APIs (DeepL, Google Translate) quickly drains free-tier allowances.
    
-   **Decision**: We utilize cross-lingual semantic representations (`LaBSE`). By using a model that understands English and German in the same high-dimensional vector space, we do not need to translate the _entire database_ to perform searches. We only need a translation step for the _summarized output_ shown to the user, keeping inference API usage within the free, rate-limited tier.
