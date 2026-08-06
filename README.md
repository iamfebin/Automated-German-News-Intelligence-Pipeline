---
title: Multilingual German News Intelligence
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
# Multilingual German News Intelligence Platform

A zero-operational-cost, production-grade NLP pipeline and search application designed to ingest, process, embed, index, and monitor German-language news media. It enables English-speaking users to query regional German media semantically in English, returning contextual multilingual summaries, localized Named Entity Recognition (NER), and an automated MLOps data drift dashboard.

## System Architecture

The platform operates in two main phases:
1.  **Batch ETL Pipeline (GitHub Actions)**:
    -   Scrapes RSS feeds from **Tagesschau**, **Deutsche Welle**, and **Der Spiegel**.
    -   Extracts full article body text and normalizes Unicode/HTML tags.
    -   Computes German Named Entity Recognition (NER) and sentence embeddings locally.
    -   Generates a translated summary of each article using a translation model.
    -   Performs statistical data drift analysis with Evidently AI (embeddings & text properties).
    -   Commits Parquet metadata, the FAISS index, and the Evidently HTML report to a Hugging Face Dataset repository.
2.  **Web Portal (FastAPI + HTML/CSS/JS)**:
    -   Loads the latest Parquet metadata and FAISS index from Hugging Face on start.
    -   Accepts English/German queries, embedding them on-demand via the HF Serverless Inference API.
    -   Performs L2/Cosine similarity vector search and displays the retrieved articles, including extracted entity badges grouped by category, German text, and English summaries.
    -   Embeds the Evidently AI HTML report under the "Ingestion Health" tab.

---

## Getting Started

### Local Setup (Zero-Config Development Mode)

If no Hugging Face credentials are set, the application automatically functions in **offline local mode**, saving and reading data from a local `data/` directory.

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd Automated-German-News-Intelligence-Pipeline
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate      # On Windows
    source .venv/bin/activate    # On Unix/macOS
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements_etl.txt
    pip install -r requirements.txt
    ```

4.  **Run the ETL Pipeline manually**:
    ```bash
    python src/job_etl.py
    ```
    *This will scrape articles, extract entities, compute embeddings, build the FAISS index, compute drift, and save them in `data/`.*

5.  **Start the Web Application (FastAPI)**:
    ```bash
    python main.py
    ```

### Production Setup & Deployment (Hugging Face Spaces)

This application is configured for direct deployment on **Hugging Face Spaces**.

#### 1. Deploy the Web Portal to HF Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Set the SDK to **Docker**.
3. Push this repository's code to your Space's Git repository (or sync it via GitHub). The YAML metadata block at the top of this `README.md` will automatically build the Docker image and launch the FastAPI app.

#### 2. Configure Environment Variables and Secrets
Under your Hugging Face Space's **Settings > Variables and secrets**, add the following:
*   **Secret** `HF_TOKEN`: Your Hugging Face API access token.
*   **Variable** `HF_REPO_ID`: Your Hugging Face Dataset repository name containing the Parquet data and index (e.g., `username/german-news-intelligence`).

#### 3. Configure GitHub Actions Ingestion Secrets
To enable the daily scraping ETL job, add the following secrets in your GitHub repository's **Settings > Secrets and variables > Actions**:
*   `HF_WRITE_TOKEN`: Your Hugging Face API access token (with `write` permission).
*   `HF_REPO_ID`: Your Hugging Face Dataset repository name.

---

### Docker & Docker Compose Setup

A Docker-based setup is available to run the services in isolated environments.

1.  **Build and Start the Web Application**:
    ```bash
    docker compose up --build -d
    ```
    *This builds the Docker image and starts the web application in detached mode on [http://localhost:7860](http://localhost:7860).*

2.  **Verify the Application is Running**:
    ```bash
    docker compose ps
    ```

3.  **Run the ETL Pipeline manually in Docker**:
    To trigger scrapers, Named Entity Recognition, translation, and embedding generation inside the container:
    ```bash
    docker compose run --rm etl
    ```
    *Note: Running the ETL pipeline locally in Docker will download Hugging Face model weights inside the container. Ensure you have configured environment variables if pushing results to Hugging Face, or let it run in local-offline mode (results will be persisted in your local `./data` directory mapped via volumes).*

4.  **Configure Environment Variables (Optional)**:
    You can define `HF_TOKEN`, `HF_WRITE_TOKEN`, and `HF_REPO_ID` in a `.env` file in the project root. Docker Compose will automatically inject them:
    ```env
    HF_TOKEN=your_hf_read_token
    HF_WRITE_TOKEN=your_hf_write_token
    HF_REPO_ID=your_username/your_dataset_repo
    ```

5.  **Shutdown Services**:
    ```bash
    docker compose down
    ```

