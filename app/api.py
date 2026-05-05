"""
ARC AI — FastAPI backend.

Endpoints:
  GET  /                      → serves the chat UI
  GET  /api/health            → readiness
  GET  /api/models            → list available Ollama models
  POST /api/chat              → streaming chat response (NDJSON)
  POST /api/nlp/analyze       → run NLP analyses on a query
  POST /api/metrics/analyze   → compute citation, grounding, and ranking metrics
  POST /api/compare           → run query against all available models, return comparison
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import list_available_models, stream_chat
from app.vectorstore import VectorStore


# ---------- request / response models ----------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    model: str | None = None


class ChatSource(BaseModel):
    n: int
    source: str
    title: str
    url: str


class NLPRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    techniques: list[str] | None = None


class MetricsRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(..., min_length=1)


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    models: list[str] | None = None  # None = all available


# ---------- app ----------
app = FastAPI(title="ARC AI", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SESSIONS: dict[str, list[dict]] = {}
_MAX_HISTORY_MESSAGES = 10

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _get_all_chunks_metadata(store: VectorStore) -> list[dict]:
    """Get metadata for every chunk in the corpus (for recall denominator)."""
    try:
        all_data = store.collection.get()
        return all_data.get("metadatas") or []
    except Exception:
        return []


# ---------- health / models ----------
@app.get("/api/health")
def health():
    store = _get_store()
    return {
        "status": "ok",
        "collection_size": store.count(),
        "default_model": settings.ollama_model,
    }


@app.get("/api/models")
def models():
    return {"models": list_available_models(), "default": settings.ollama_model}


# ---------- chat ----------
@app.post("/api/chat")
async def chat(req: ChatRequest):
    store = _get_store()
    if store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base empty. Run: python -m scripts.ingest",
        )

    session_id = req.session_id or str(uuid.uuid4())
    history = _SESSIONS.setdefault(session_id, [])

    hits = store.query(req.message)

    seen_urls: set[str] = set()
    unique_sources = []
    for i, h in enumerate(hits):
        if h.url in seen_urls:
            continue
        seen_urls.add(h.url)
        unique_sources.append(
            ChatSource(n=i + 1, source=h.source, title=h.title or h.url, url=h.url).model_dump()
        )

    async def event_stream() -> AsyncIterator[bytes]:
        yield (
            json.dumps({
                "type": "meta",
                "session_id": session_id,
                "sources": unique_sources,
                "model": req.model or settings.ollama_model,
            }) + "\n"
        ).encode()

        answer_parts: list[str] = []
        try:
            for token in stream_chat(req.message, hits, history=history, model=req.model):
                answer_parts.append(token)
                yield (json.dumps({"type": "token", "text": token}) + "\n").encode()
        except Exception as e:
            yield (json.dumps({"type": "error", "error": str(e)}) + "\n").encode()
            return

        full_answer = "".join(answer_parts)
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": full_answer})
        if len(history) > _MAX_HISTORY_MESSAGES:
            del history[: len(history) - _MAX_HISTORY_MESSAGES]

        yield (json.dumps({"type": "done"}) + "\n").encode()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str):
    _SESSIONS.pop(session_id, None)
    return {"ok": True}


# ---------- NLP analyze ----------
@app.post("/api/nlp/analyze")
def nlp_analyze(req: NLPRequest):
    from app.nlp import (
        intent_classify,
        extract_entities,
        extractive_qa,
        sentiment_analysis,
        summarize_text,
        extract_keywords,
        readability_score,
        emotion_detect,
        topic_model,
    )

    store = _get_store()
    if store.count() == 0:
        raise HTTPException(status_code=503, detail="Knowledge base empty.")

    hits = store.query(req.query, k=5)
    context = " ".join(h.text for h in hits)

    TECHNIQUE_MAP = {
        "intent": lambda: intent_classify(req.query),
        "ner": lambda: extract_entities(context),
        "extractive_qa": lambda: extractive_qa(req.query, context),
        "sentiment": lambda: sentiment_analysis(context),
        "summary": lambda: summarize_text(context),
        "keywords": lambda: extract_keywords(context),
        "readability": lambda: readability_score(context),
        "emotion": lambda: emotion_detect(context),
        "topics": lambda: topic_model([c.text for c in store.query(req.query, k=store.count())])
                          if store.count() >= 5 else {"note": "not enough chunks"},
    }

    if req.techniques:
        run = {k: v for k, v in TECHNIQUE_MAP.items() if k in req.techniques}
    else:
        run = {k: v for k, v in TECHNIQUE_MAP.items() if k != "topics"}

    results = {}
    for name, fn in run.items():
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "query": req.query,
        "n_techniques": len(results),
        "results": results,
    }


# ---------- Metrics analyze ----------
@app.post("/api/metrics/analyze")
def metrics_analyze(req: MetricsRequest):
    """Compute runtime metrics for a given (query, answer) pair."""
    from app.metrics import compute_metrics

    store = _get_store()
    if store.count() == 0:
        raise HTTPException(status_code=503, detail="Knowledge base empty.")

    hits = store.query(req.query, k=5)
    retrieved_chunks = [
        {
            "text": h.text,
            "url": h.url,
            "source": h.source,
            "title": h.title,
            "similarity": h.similarity,
        }
        for h in hits
    ]
    all_metadata = _get_all_chunks_metadata(store)

    metrics = compute_metrics(
        req.answer, retrieved_chunks, req.query, all_chunks_metadata=all_metadata
    )

    return {
        "query": req.query,
        "metrics": metrics,
    }


# ---------- Multi-model comparison ----------
@app.post("/api/compare")
def compare_models(req: CompareRequest):
    """
    Run a single query against multiple LLMs and return a comparison
    table of metrics for each. Used for evaluating model trade-offs.
    """
    from app.metrics import compute_metrics

    store = _get_store()
    if store.count() == 0:
        raise HTTPException(status_code=503, detail="Knowledge base empty.")

    # Determine which models to compare
    available = list_available_models()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No Ollama models available. Pull at least one model.",
        )

    if req.models:
        # Filter to requested models that are actually available
        models_to_run = [m for m in req.models if m in available]
        if not models_to_run:
            raise HTTPException(
                status_code=400,
                detail=f"None of the requested models are available. Available: {available}",
            )
    else:
        models_to_run = available

    # Retrieve once; same context for all models
    hits = store.query(req.query, k=5)
    retrieved_chunks = [
        {
            "text": h.text,
            "url": h.url,
            "source": h.source,
            "title": h.title,
            "similarity": h.similarity,
        }
        for h in hits
    ]
    all_metadata = _get_all_chunks_metadata(store)

    # Run each model
    per_model = []
    for model_name in models_to_run:
        t0 = time.time()
        try:
            tokens = list(stream_chat(req.query, hits, history=[], model=model_name))
            answer = "".join(tokens)
            generation_time = time.time() - t0

            # Compute metrics for this answer
            metrics = compute_metrics(
                answer, retrieved_chunks, req.query, all_chunks_metadata=all_metadata
            )

            # Build a compact summary for the comparison table
            summary = {
                "model": model_name,
                "answer": answer,
                "answer_length_chars": len(answer),
                "answer_length_words": metrics["grounding_metrics"]["answer_word_count"],
                "generation_time_sec": round(generation_time, 2),
                "citation_precision": metrics["citation_metrics"]["citation_precision"],
                "citation_coverage": metrics["citation_metrics"]["citation_coverage"],
                "total_citations": metrics["citation_metrics"]["total_citations"],
                "hallucinated_citations": metrics["citation_metrics"]["hallucinated_citations"],
                "answer_context_overlap": metrics["grounding_metrics"]["answer_context_overlap"],
                "precision_at_k": metrics["ranking_metrics"]["precision_at_k"],
                "recall_at_k": metrics["ranking_metrics"]["recall_at_k"],
                "mrr": metrics["ranking_metrics"]["mrr"],
                "ndcg_at_k": metrics["ranking_metrics"]["ndcg_at_k"],
                "evaluation_mode": metrics["ranking_metrics"]["evaluation_mode"],
                "topic_label": metrics["ranking_metrics"]["topic_label"],
                "interpretation": metrics["interpretation"],
                "error": None,
            }
        except Exception as e:
            summary = {
                "model": model_name,
                "answer": "",
                "error": str(e),
            }
        per_model.append(summary)

    return {
        "query": req.query,
        "n_models": len(per_model),
        "n_chunks_retrieved": len(retrieved_chunks),
        "models_compared": [m["model"] for m in per_model],
        "results": per_model,
    }


# ---------- static frontend ----------
_STATIC_DIR = Path(__file__).parent.parent / "web"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    def root():
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="UI not built")