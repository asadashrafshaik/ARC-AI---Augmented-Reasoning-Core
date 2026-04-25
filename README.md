# 🏠 ARC AI — Maryland Housing & Rental Law Assistant

> **A local-first, retrieval-augmented chatbot that answers Maryland housing and rental law questions with transparent citations — powered by Llama 3.1, ChromaDB, and 9 NLP analysis techniques.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-local_LLM-black.svg)](https://ollama.com)


---

## Overview

ARC AI scrapes official Maryland state and county housing law sources, embeds the content into a local vector database, and uses a local LLM to answer tenant/landlord questions with inline citations back to the original source URLs. No data leaves your machine.

The system includes a custom-designed web UI with multi-model switching (Llama 3.1 / Mistral / Qwen 2.5), streaming responses, conversation history, and an integrated NLP analysis panel that runs 8 techniques on every answer at the click of a button.

### Key Features

- **Retrieval-Augmented Generation (RAG)** — answers grounded in real Maryland legal sources, not hallucinated
- **Multi-model chat** — switch between Llama 3.1 8B, Mistral 7B, and Qwen 2.5 7B mid-conversation
- **9 NLP techniques** — intent classification, NER, extractive QA, sentiment analysis, emotion detection, summarization, keyword extraction, readability scoring, and topic modeling
- **Live web scraping** — pulls from 7 official Maryland sources (state + county level)
- **Custom UI** — warm beige/terracotta theme with streaming responses, citation pills, and an expandable NLP analysis panel
- **Fully local** — runs entirely on your machine via Ollama; no API keys, no cloud dependencies

---

## Architecture

The system follows a Retrieval-Augmented Generation pipeline: user questions are embedded with MiniLM, matched against chunked legal documents in ChromaDB via cosine similarity, and the top-K results are injected into a grounded prompt sent to a local LLM (Ollama). The response streams back with inline citations. An optional NLP analysis layer runs 9 techniques (intent classification, NER, extractive QA, sentiment, emotion, summarization, keywords, readability, topic modeling) on the retrieved context.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Ollama — Llama 3.1 8B (default), Mistral 7B, Qwen 2.5 7B |
| **Embeddings** | sentence-transformers / all-MiniLM-L6-v2 (384-dim) |
| **Vector store** | ChromaDB (persistent, cosine similarity) |
| **Scraper** | requests + BeautifulSoup4 + lxml |
| **Backend** | FastAPI (streaming NDJSON, session memory) |
| **Frontend** | Custom HTML/CSS/JS — Fraunces + Inter + JetBrains Mono |
| **NLP** | spaCy, BART-MNLI, RoBERTa-SQuAD2, BART-CNN, KeyBERT, VADER, distilRoBERTa |
| **Language** | Python 3.11 |

---

## Data Sources

All sources are scraped once and stored locally. The scraper uses a browser User-Agent and keyword-filtered 2-hop crawling.

| Source | Coverage |
|---|---|
| Maryland Attorney General | Landlord/tenant disputes guide |
| Maryland DHCD | State Office of Tenant & Landlord Affairs |
| Montgomery County DHCA | County-level handbook, tenant rights, landlord responsibilities |
| Baltimore County | Circuit Court Law Library, housing resources |
| Baltimore City DHCD | Resources for renters |
| Prince George's County DHCD | Tenant resources, Rent Stabilization Act 2024 |
| People's Law Library | Community legal resource (landlord-tenant, housing) |

---

## NLP Analysis Pipeline

Each technique is accessible via the "Analyze (NLP)" button in the UI or the POST /api/nlp/analyze endpoint.

| # | Technique | Model | Purpose |
|---|---|---|---|
| 1 | Intent classification | facebook/bart-large-mnli | Route queries: deposit vs eviction vs repairs |
| 2 | Named Entity Recognition | spaCy en_core_web_sm + regex | Extract dates, dollar amounts, statute refs |
| 3 | Topic modeling | sklearn/LDA | Discover themes across the corpus |
| 4 | Extractive QA | deepset/roberta-base-squad2 | Pull exact answer span for verification |
| 5 | Sentiment analysis | VADER + cardiffnlp/roberta-sentiment | Detect tone of legal text |
| 6 | Text summarization | facebook/bart-large-cnn | TL;DR of retrieved legal passages |
| 7 | Keyword extraction | KeyBERT/all-MiniLM-L6-v2 | Key legal terms with relevance scores |
| 8 | Readability scoring | Flesch-Kincaid (pure Python) | Grade level and reading difficulty |
| 9 | Emotion detection | j-hartmann/distilroberta-emotion | Emotional tone classification |

---

## Quick Start

### Prerequisites

- **Python 3.11** (not 3.14 — binary wheels unavailable for key packages)
- **Ollama** — https://ollama.com/download
- **Git** — https://git-scm.com

### 1. Clone

```bash
git clone https://github.com/mafaisalhussain/ARC-AI-Augmented-Reasoning-Core.git
cd ARC-AI-Augmented-Reasoning-Core
```

### 2. Virtual environment

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 4. Pull models

```bash
ollama pull llama3.1:8b       # required — default model
ollama pull mistral:7b         # optional — for model switching
ollama pull qwen2.5:7b         # optional — for model switching
```

### 5. Configure

```bash
cp .env.example .env           # Linux/macOS
Copy-Item .env.example .env    # Windows PowerShell
```

### 6. Ingest (scrape, chunk, embed)

```bash
python -m scripts.ingest
```

Scrapes 7 Maryland sources, chunks into ~500-token pieces, embeds with MiniLM, stores in ChromaDB. Takes 3–6 minutes (one-time).

### 7. Launch

```bash
python -m scripts.serve
```

Open **http://127.0.0.1:8000**.

---

## Usage

### Web UI

- Type a question or click a suggestion on the welcome screen
- Responses stream token-by-token with [S1], [S2] inline citations
- Click source pills below answers to open the original Maryland source
- Switch models via the Model dropdown (top right)
- Click "Analyze (NLP)" under any answer to expand the 8-technique analysis panel
- Click "New conversation" to start fresh

### CLI Chat

```bash
python -m app.chat
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Web UI |
| GET | /api/health | Readiness check + collection size |
| GET | /api/models | Available Ollama models |
| POST | /api/chat | Streaming chat (NDJSON) |
| POST | /api/nlp/analyze | NLP analysis (8 techniques) |
| DELETE | /api/sessions/{id} | Clear session history |

### NLP Analysis Example

```bash
curl -X POST http://127.0.0.1:8000/api/nlp/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the maximum security deposit in Maryland?"}'
```

Run specific techniques only:

```bash
curl -X POST http://127.0.0.1:8000/api/nlp/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Can my landlord evict me?", "techniques": ["intent", "extractive_qa"]}'
```

---

## Project Structure

```
├── app/
│   ├── config.py           # Central settings (reads .env)
│   ├── scraper.py          # Multi-source scraper (browser UA, 2-hop crawl)
│   ├── chunker.py          # Token-aware chunking (tiktoken)
│   ├── vectorstore.py      # ChromaDB wrapper (MiniLM embeddings)
│   ├── llm.py              # Ollama client (streaming, multi-model)
│   ├── nlp.py              # 9 NLP techniques (lazy-loaded models)
│   ├── chat.py             # CLI REPL
│   └── api.py              # FastAPI backend
├── scripts/
│   ├── ingest.py           # Scrape → chunk → embed → store
│   └── serve.py            # Server launcher
├── web/
│   └── index.html          # Custom chat UI
├── data/
│   ├── raw/                # Scraped JSONL (git-ignored)
│   └── processed/          # ChromaDB (git-ignored)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
## Configuration

All settings via .env (copied from .env.example):

| Variable | Default | Description |
|---|---|---|
| OLLAMA_HOST | http://localhost:11434 | Ollama server URL |
| OLLAMA_MODEL | llama3.1:8b | Default chat model |
| EMBED_MODEL | all-MiniLM-L6-v2 | Embedding model |
| TOP_K | 5 | Chunks retrieved per query |
| CHUNK_SIZE | 500 | Tokens per chunk |
| CHUNK_OVERLAP | 75 | Token overlap between chunks |
| REQUEST_DELAY | 1.0 | Seconds between scraper requests |

---

## Limitations

- **Coverage:** Some state-level sources use heavy JavaScript or Cloudflare protection, resulting in sparse scraped content. A production system would use cloudscraper or headless Chromium.
- **Freshness:** Point-in-time scrape snapshot. Maryland law changes annually. Scheduled re-scraping needed for production.
- **Hallucination:** Despite retrieval grounding, the LLM can occasionally paraphrase context misleadingly. The extractive QA fallback mitigates this.
- **Hardware:** Llama 3.1 8B requires ~5 GB RAM. CPU-only response times: 10–30 seconds.

---

## Future Work

- County-specific retrieval routing based on user location mentions
- QLoRA fine-tuning on Maryland legal QA pairs
- PDF upload: answer questions grounded in the user's lease + state law corpus
- Cloud deployment (Render/Fly.io + RunPod for hosted Ollama)

---

## Disclaimer

**ARC AI is not a lawyer.** All answers are informational only, based on publicly available Maryland state and county sources. Consult a licensed attorney for advice specific to your situation.

---


