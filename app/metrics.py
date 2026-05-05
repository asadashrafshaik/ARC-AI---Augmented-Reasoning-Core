"""
Runtime retrieval and citation metrics for ARC-AI.

For each generated answer, we compute:
  - Citation precision: fraction of [Sn] markers backed by the cited chunk
  - Citation coverage: fraction of factual sentences that include a citation
  - Hallucinated citation count: [Sn] markers pointing to non-existent sources
  - Retrieval cohesion: source / URL diversity in top-K
  - Answer-context overlap: fraction of answer noun phrases present in context
  - Retrieval P@K, R@K, MRR, nDCG@K (Option C — hybrid)
        For queries matching a known housing topic, evaluated against
        URL-keyword labels (real evaluation).
        For other queries, fall back to similarity-threshold self-consistency
        (clearly labeled as such in the response).
"""
from __future__ import annotations

import re
import math


# ─── Known-topic relevance patterns ──────────────────────────────────────────
# Each entry: (topic_name, query_keywords, relevant_url_keywords)
# A query matches the topic if it contains ANY query_keyword.
# A retrieved chunk URL is relevant if it contains ANY relevant_url_keyword.

KNOWN_TOPICS = [
    {
        "topic": "security_deposit",
        "query_keywords": ["security deposit", "deposit"],
        "relevant_url_keywords": ["security-deposit", "tenant-rights", "landlord-tenant", "handbook"],
    },
    {
        "topic": "repairs_habitability",
        "query_keywords": ["repair", "fix", "broken", "habitability", "maintenance", "code"],
        "relevant_url_keywords": ["repair", "habitability", "code-enforcement", "tenant-rights", "handbook"],
    },
    {
        "topic": "eviction",
        "query_keywords": ["evict", "eviction", "kick out", "removal"],
        "relevant_url_keywords": ["eviction", "failure-to-pay", "rent", "court"],
    },
    {
        "topic": "rent_increase",
        "query_keywords": ["rent increase", "raise rent", "rent hike", "raising rent"],
        "relevant_url_keywords": ["rent-increase", "lease", "notice", "handbook"],
    },
    {
        "topic": "landlord_entry",
        "query_keywords": ["enter", "entry", "privacy", "without permission"],
        "relevant_url_keywords": ["entry", "privacy", "tenant-rights", "handbook"],
    },
    {
        "topic": "lease_termination",
        "query_keywords": ["lease termination", "end lease", "break lease", "terminate"],
        "relevant_url_keywords": ["lease", "termination", "notice", "handbook"],
    },
    {
        "topic": "discrimination",
        "query_keywords": ["discriminate", "discrimination", "fair housing"],
        "relevant_url_keywords": ["discrimination", "fair-housing", "landlord-tenant"],
    },
    {
        "topic": "rental_application",
        "query_keywords": ["application", "apply", "rental agreement", "lease agreement"],
        "relevant_url_keywords": ["lease", "rental-agreement", "handbook", "application"],
    },
    {
        "topic": "tenant_rights_general",
        "query_keywords": ["my rights", "tenant rights", "what are my rights"],
        "relevant_url_keywords": ["tenant-rights", "landlord-tenant", "handbook"],
    },
    {
        "topic": "office_landlord_tenant",
        "query_keywords": ["office of landlord", "complaint", "where to complain", "mediation"],
        "relevant_url_keywords": ["landlord-tenant", "office", "affairs", "complaint"],
    },
]


# ─── Citation analysis (existing) ────────────────────────────────────────────

def extract_citations(answer_text: str) -> list[int]:
    """Extract [S1], [S2], ... citation indices from answer text."""
    return [int(m) for m in re.findall(r"\[S(\d+)\]", answer_text)]


def split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter for citation analysis."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_simple(text: str) -> set[str]:
    """Lowercase word tokenization for overlap computation."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


# ─── Retrieval metrics (Option C — hybrid) ───────────────────────────────────

def find_topic(query: str) -> dict | None:
    """Return the known topic for a query, or None if no match."""
    q_lower = query.lower()
    for topic in KNOWN_TOPICS:
        if any(kw.lower() in q_lower for kw in topic["query_keywords"]):
            return topic
    return None


def is_url_relevant(chunk_url: str, relevant_url_keywords: list[str]) -> bool:
    """A chunk URL is relevant if it contains any relevant keyword."""
    if not relevant_url_keywords:
        return False
    url_lower = chunk_url.lower()
    return any(kw.lower() in url_lower for kw in relevant_url_keywords)


def precision_at_k(retrieved_urls: list[str], relevant_keywords: list[str], k: int) -> float:
    """P@K = (relevant items in top K) / K."""
    if k == 0:
        return 0.0
    top_k = retrieved_urls[:k]
    rel = sum(1 for url in top_k if is_url_relevant(url, relevant_keywords))
    return rel / k


def recall_at_k(retrieved_urls: list[str], relevant_keywords: list[str], k: int,
                total_relevant_in_corpus: int) -> float:
    """R@K = (relevant items in top K) / total relevant in corpus."""
    if total_relevant_in_corpus == 0:
        return 0.0
    top_k = retrieved_urls[:k]
    rel = sum(1 for url in top_k if is_url_relevant(url, relevant_keywords))
    return rel / total_relevant_in_corpus


def reciprocal_rank(retrieved_urls: list[str], relevant_keywords: list[str]) -> float:
    """RR = 1 / rank of first relevant item, or 0 if none found."""
    for i, url in enumerate(retrieved_urls):
        if is_url_relevant(url, relevant_keywords):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_urls: list[str], relevant_keywords: list[str], k: int) -> float:
    """nDCG@K with binary relevance."""
    top_k = retrieved_urls[:k]
    rels = [1 if is_url_relevant(url, relevant_keywords) else 0 for url in top_k]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    n_rel = sum(rels)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(query: str,
                              retrieved_chunks: list[dict],
                              all_chunks_metadata: list[dict],
                              k: int = 5) -> dict:
    """
    Compute retrieval metrics using hybrid approach:
      - If query matches a known topic, use URL keyword relevance (real eval)
      - Otherwise, use similarity threshold self-consistency (fallback)
    """
    topic = find_topic(query)
    retrieved_urls = [c.get("url", "") for c in retrieved_chunks]

    if topic is None:
        # ───── Fallback: similarity-threshold self-consistency ─────
        # Use cosine similarity stored on chunks (if available)
        # Treat distance < 0.5 as "relevant" (cosine_sim > 0.5)
        relevant_in_topk = 0
        for c in retrieved_chunks:
            sim = c.get("similarity", None)
            if sim is not None and sim >= 0.5:
                relevant_in_topk += 1

        return {
            "evaluation_mode": "self_consistency",
            "topic": None,
            "topic_label": "Out of evaluation set",
            "note": "Query did not match any pre-defined housing topic. "
                    "Reported numbers use similarity-threshold relevance "
                    "(chunks with cosine similarity ≥ 0.5 treated as relevant). "
                    "Treat with caution — this is a self-consistency check, not "
                    "ground-truth evaluation.",
            "precision_at_k": round(relevant_in_topk / k, 3) if k > 0 else 0.0,
            "recall_at_k": None,  # cannot compute without total_relevant
            "mrr": None,
            "ndcg_at_k": None,
            "k": k,
            "n_relevant_in_topk": relevant_in_topk,
            "n_relevant_in_corpus": None,
        }

    # ───── Real eval: URL keyword labels ─────
    rel_kws = topic["relevant_url_keywords"]

    # Count total relevant chunks in the entire corpus
    total_relevant = sum(
        1 for meta in all_chunks_metadata
        if is_url_relevant(meta.get("url", ""), rel_kws)
    )

    p_at_k = precision_at_k(retrieved_urls, rel_kws, k)
    r_at_k = recall_at_k(retrieved_urls, rel_kws, k, total_relevant)
    rr = reciprocal_rank(retrieved_urls, rel_kws)
    ndcg = ndcg_at_k(retrieved_urls, rel_kws, k)

    n_relevant_in_topk = sum(1 for url in retrieved_urls[:k] if is_url_relevant(url, rel_kws))

    return {
        "evaluation_mode": "ground_truth",
        "topic": topic["topic"],
        "topic_label": topic["topic"].replace("_", " ").title(),
        "note": "Query matched a known housing topic. Relevance determined by "
                "URL keyword labels (real evaluation against pre-defined gold set).",
        "precision_at_k": round(p_at_k, 3),
        "recall_at_k": round(r_at_k, 3),
        "mrr": round(rr, 3),
        "ndcg_at_k": round(ndcg, 3),
        "k": k,
        "n_relevant_in_topk": n_relevant_in_topk,
        "n_relevant_in_corpus": total_relevant,
    }


# ─── Main entry point ────────────────────────────────────────────────────────

def compute_metrics(answer_text: str,
                    retrieved_chunks: list[dict],
                    query: str,
                    all_chunks_metadata: list[dict] | None = None) -> dict:
    """
    Compute all runtime metrics for a single answer.

    Args:
        answer_text: the full LLM-generated answer
        retrieved_chunks: list of {text, url, source, title, similarity?} for top-K hits
        query: the user's original query
        all_chunks_metadata: list of all chunk metadata in the corpus (for recall denominator).
                             If None, recall computation is skipped.

    Returns:
        dict with all computed metrics, ready to JSON-serialize
    """
    # ---------- Citation analysis ----------
    citations = extract_citations(answer_text)
    n_chunks = len(retrieved_chunks)

    valid_citations = [c for c in citations if 1 <= c <= n_chunks]
    hallucinated_citations = [c for c in citations if c < 1 or c > n_chunks]

    supported_count = 0
    sentences = split_into_sentences(answer_text)
    cited_sentences = [s for s in sentences if re.search(r"\[S\d+\]", s)]

    for sentence in cited_sentences:
        cite_indices = extract_citations(sentence)
        sentence_tokens = tokenize_simple(re.sub(r"\[S\d+\]", "", sentence))
        for idx in cite_indices:
            if 1 <= idx <= n_chunks:
                chunk_tokens = tokenize_simple(retrieved_chunks[idx - 1].get("text", ""))
                if sentence_tokens:
                    overlap = len(sentence_tokens & chunk_tokens) / len(sentence_tokens)
                    if overlap >= 0.30:
                        supported_count += 1

    citation_precision = (
        supported_count / len(valid_citations) if valid_citations else 0.0
    )

    factual_sentences = [s for s in sentences if len(s) > 40]
    cited_factual = [s for s in factual_sentences if re.search(r"\[S\d+\]", s)]
    citation_coverage = (
        len(cited_factual) / len(factual_sentences) if factual_sentences else 0.0
    )

    # ---------- Retrieval analysis ----------
    unique_sources = set()
    unique_urls = set()
    for c in retrieved_chunks:
        unique_sources.add(c.get("source", ""))
        unique_urls.add(c.get("url", ""))

    # ---------- Grounding analysis ----------
    answer_tokens = tokenize_simple(answer_text)
    context_tokens = set()
    for c in retrieved_chunks:
        context_tokens |= tokenize_simple(c.get("text", ""))

    if answer_tokens:
        grounding_overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    else:
        grounding_overlap = 0.0

    query_tokens = tokenize_simple(query)
    if query_tokens:
        query_in_answer = len(query_tokens & answer_tokens) / len(query_tokens)
    else:
        query_in_answer = 0.0

    # ---------- Retrieval metrics (P@K, R@K, MRR, nDCG) ----------
    retrieval_eval = compute_retrieval_metrics(
        query, retrieved_chunks, all_chunks_metadata or [], k=5
    )

    return {
        "citation_metrics": {
            "total_citations": len(citations),
            "valid_citations": len(valid_citations),
            "supported_citations": supported_count,
            "hallucinated_citations": len(hallucinated_citations),
            "citation_precision": round(citation_precision, 3),
            "citation_coverage": round(citation_coverage, 3),
            "n_factual_sentences": len(factual_sentences),
            "n_cited_sentences": len(cited_factual),
        },
        "retrieval_metrics": {
            "n_chunks_retrieved": n_chunks,
            "unique_sources": len(unique_sources),
            "unique_urls": len(unique_urls),
            "sources_used": sorted(unique_sources),
        },
        "ranking_metrics": retrieval_eval,
        "grounding_metrics": {
            "answer_context_overlap": round(grounding_overlap, 3),
            "query_in_answer": round(query_in_answer, 3),
            "answer_word_count": len(answer_tokens),
            "context_word_count": len(context_tokens),
        },
        "interpretation": _interpret(citation_precision, citation_coverage, grounding_overlap),
    }


def _interpret(cite_prec: float, cite_cov: float, grounding: float) -> dict:
    """Human-readable assessment for the UI."""
    def label(score: float, low: float, high: float) -> str:
        if score >= high:
            return "strong"
        elif score >= low:
            return "moderate"
        else:
            return "weak"

    return {
        "citation_quality": label(cite_prec, 0.5, 0.8),
        "citation_thoroughness": label(cite_cov, 0.4, 0.7),
        "grounding_strength": label(grounding, 0.4, 0.7),
    }