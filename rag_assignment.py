"""
Improving RAG with reranking, chunking, and conversational memory.

Corpus
    Hugging Face `ccdv/arxiv-summarization`, config `document`,
    train split, first 100 papers. Articles are full papers, not
    pre-chunked. Math and citation placeholders (@xmath, @xcite) are
    normalized during preprocessing.

What this script does
    1. Preprocess the 100 papers.
    2. Chunk them two ways:
         - fixed_window: a sliding character window (cuts mid-sentence)
         - sentence_packed: packs whole sentences up to the same budget
    3. Embed each chunk with all-MiniLM-L6-v2 and store both indexes in
       a local ChromaDB collection (cosine similarity).
    4. Retrieve the same queries with the bi-encoder alone, then again
       after a cross-encoder reranks the top candidates.
    5. Demonstrate conversational memory on a two-turn question whose
       second turn only makes sense given the first.
    6. Score every run with Precision@5, Recall@5, and MRR.

Relevance
    Labels are manual. Abstracts were read and each query was assigned
    the paper that actually answers it. A chunk from that paper is
    marked useful when it also contains the answer-bearing phrases
    listed in QUERIES / MEMORY. The resulting chunk ids are written to
    results/relevance_judgments.json.

Run
    .venv\\Scripts\\python.exe rag_assignment.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "arxiv_100.jsonl"
CHROMA_PATH = ROOT / "chroma_db"
RESULTS_DIR = ROOT / "results"

N_DOCS = 100
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 180
TOP_K = 5
CANDIDATE_POOL = 20
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_ADD = 250

# Hand labels. gold_docs are paper indexes in the 100-document subset.
# A chunk is relevant when it belongs to one of those papers AND at least
# `min_hits` of `keywords` occur in it. `require_all` phrases are mandatory.
QUERIES = [
    {
        "id": "q1_additive_kernels",
        "text": "What learning rates are obtained for regularized kernel methods applied to additive models?",
        "gold_docs": [0],
        "keywords": ["learning rate", "kernel", "additive"],
        "min_hits": 3,
        "why": "Document 0 derives those learning rates. A chunk is useful only when it actually discusses learning rates, kernels, and additive models together, not when it is a generic introduction.",
    },
    {
        "id": "q2_leptonic_decay",
        "text": "How is a leptonic decay constant extracted from tagged meson decays?",
        "gold_docs": [1],
        "keywords": ["decay constant", "leptonic"],
        "min_hits": 2,
        "why": "Document 1 extracts a leptonic decay constant from tagged decays. The query does not name CLEO-c. Other papers mention a decay constant in passing, so those chunks are not labeled relevant.",
    },
    {
        "id": "q3_synaptic_memory",
        "text": "How do nanoelectronic synapses store synaptic memory for neuromorphic computing?",
        "gold_docs": [7],
        "keywords": ["mtj", "plasticity"],
        "min_hits": 2,
        "why": "Document 7's answer is that an MTJ's magnetization dynamics reproduce synaptic plasticity. Background chunks about brain power are not useful even though they come from the same paper.",
    },
    {
        "id": "q4_od_matrix",
        "text": "How can origin-destination passenger counts be estimated for public transport?",
        "gold_docs": [32],
        "keywords": ["od matrix", "origin - destination"],
        "min_hits": 1,
        "why": "Document 32 is the only public-transport origin-destination paper in the subset. The body writes this both as 'od matrix' and as 'origin - destination'. Chunks that mention neither are not an answer.",
    },
    {
        "id": "q5_magnetoresistance",
        "text": "Why does magnetoresistance stay linear instead of saturating on a topological insulator surface?",
        "gold_docs": [20],
        "keywords": ["magnetoresistance", "surface state", "topological"],
        "min_hits": 2,
        "why": "Document 20 explains linear, non-saturating magnetoresistance on a topological insulator surface. Documents 7 and 42 also mention magnetoresistance, and documents 70 and 71 discuss topological insulators.",
    },
    {
        "id": "q6_template_bank",
        "text": "How are banks of waveform templates used to detect compact-object mergers in gravitational-wave data?",
        "gold_docs": [27],
        "keywords": ["template bank", "gravitational", "merger", "waveform"],
        "min_hits": 2,
        "why": "Document 27 builds a template bank for matched filtering of compact-object mergers. Many other papers mention gravity, so a chunk must be about the template bank or the merger search.",
    },
]

# Turn 2 is deliberately incomplete. "it" refers to the magnetic tunnel
# junction synapse introduced in turn 1. The raw follow-up is mostly about
# "frequency", which is common across this corpus. The resolved question is not.
MEMORY_TURNS = [
    {
        "id": "mem_turn1",
        "text": "How do magnetic tunnel junctions act as nanoelectronic synapses for neuromorphic memory?",
        "gold_docs": [7],
        "keywords": ["tunnel junction", "synaptic", "neuromorphic", "plasticity"],
        "min_hits": 2,
        "why": "Same paper as q3. Turn 1 names the device so later anaphora has a referent.",
    },
    {
        "id": "mem_turn2",
        "text": "How does the frequency of the input affect it?",
        "gold_docs": [7],
        "keywords": ["potentiation", "plasticity", "tunnel junction", "synaptic", "mtj"],
        "require_all": ["frequency"],
        "min_hits": 1,
        "why": "Document 7 argues that stimulus frequency decides whether the junction shows long-term potentiation. Chunks are useful only when they discuss frequency in that synaptic setting.",
    },
]

ANAPHORA = re.compile(
    r"\b(it|its|they|their|them|this|that|those|these)\b",
    re.IGNORECASE,
)


def load_documents(n: int = N_DOCS) -> list[dict]:
    """Load the first n training documents, downloading them if needed."""
    if not DATA_PATH.exists():
        from datasets import load_dataset

        print(f"Downloading first {n} documents of ccdv/arxiv-summarization (document/train)...")
        stream = load_dataset(
            "ccdv/arxiv-summarization",
            "document",
            split="train",
            streaming=True,
        )
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DATA_PATH.open("w", encoding="utf-8") as handle:
            for i, row in enumerate(stream):
                if i >= n:
                    break
                record = {
                    "doc_id": i,
                    "article": row["article"],
                    "abstract": row["abstract"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    docs = []
    with DATA_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            docs.append(json.loads(line))
            if len(docs) >= n:
                break
    if len(docs) < n:
        raise RuntimeError(f"Expected {n} documents, found {len(docs)} in {DATA_PATH}")
    return docs


def preprocess(text: str) -> str:
    """Normalize dataset placeholders and whitespace.

    The arXiv summarization release replaces math with @xmath tokens and
    citations with @xcite. Those tokens are not words, so they are mapped
    to short markers before chunking. Whitespace is collapsed because the
    release already flattens most line breaks into spaces.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"@xmath\d*", " [MATH] ", text)
    text = re.sub(r"@xcite", " [CITE] ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_fixed(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Strategy 1: sliding window over characters.

    Boundaries ignore sentence endings, so a chunk often starts or stops
    mid-sentence. Overlap keeps a fact that sits on a boundary in two chunks.
    """
    if len(text) <= size:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if len(piece) >= MIN_CHUNK_CHARS:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:
            break
        start = next_start
    return chunks


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _join_length(parts: list[str]) -> int:
    if not parts:
        return 0
    return sum(len(part) for part in parts) + (len(parts) - 1)


def chunk_sentences(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Strategy 2: pack whole sentences up to the same character budget.

    A sentence longer than the budget is still split, because the embedding
    model cannot use an unbounded span. Otherwise chunks end on sentence
    boundaries. When there is room, the last sentence of a full chunk is
    repeated at the start of the next chunk so a claim and the sentence
    that follows it can land in the same passage.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []

    def emit(parts: list[str]) -> None:
        joined = " ".join(parts).strip()
        if len(joined) >= MIN_CHUNK_CHARS:
            chunks.append(joined)

    for sentence in sentences:
        if len(sentence) > size:
            if current:
                emit(current)
                current = []
            for offset in range(0, len(sentence), size):
                piece = sentence[offset : offset + size].strip()
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append(piece)
            continue

        if current and _join_length(current) + 1 + len(sentence) > size:
            emit(current)
            overlap = current[-1]
            if len(current) > 1 and len(overlap) + 1 + len(sentence) <= size:
                current = [overlap]
            else:
                current = []
        current.append(sentence)

    if current:
        emit(current)
    return chunks


def build_chunks(docs: list[dict], strategy: str) -> list[dict]:
    splitter = chunk_fixed if strategy == "fixed_window" else chunk_sentences
    records = []
    for doc in docs:
        body = preprocess(doc["article"])
        for index, piece in enumerate(splitter(body)):
            records.append(
                {
                    "id": f"{strategy}-{doc['doc_id']}-{index}",
                    "doc_id": int(doc["doc_id"]),
                    "chunk_index": index,
                    "strategy": strategy,
                    "text": piece,
                }
            )
    return records


def chunk_stats(chunks: list[dict]) -> dict:
    lengths = [len(chunk["text"]) for chunk in chunks]
    return {
        "n_chunks": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }


def is_relevant(text: str, spec: dict) -> bool:
    lowered = text.lower()
    for required in spec.get("require_all", []):
        if required not in lowered:
            return False
    hits = sum(1 for keyword in spec["keywords"] if keyword in lowered)
    return hits >= spec["min_hits"]


def relevant_id_set(chunks: list[dict], spec: dict) -> set[str]:
    gold_docs = set(spec["gold_docs"])
    return {
        chunk["id"]
        for chunk in chunks
        if chunk["doc_id"] in gold_docs and is_relevant(chunk["text"], spec)
    }


def embed_chunks(model: SentenceTransformer, chunks: list[dict], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) == len(chunks):
            print(f"Reusing cached embeddings at {cache_path.name}")
            return cached
        print(f"Cache {cache_path.name} has {len(cached)} rows, expected {len(chunks)}. Recomputing.")

    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL_NAME}...")
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors


def index_chunks(collection_name: str, chunks: list[dict], embeddings: np.ndarray) -> chromadb.Collection:
    """Create or replace a persistent Chroma collection of precomputed embeddings."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    existing = {col.name for col in client.list_collections()}
    if collection_name in existing:
        current = client.get_collection(collection_name)
        if current.count() == len(chunks):
            print(f"Reusing Chroma collection '{collection_name}' ({current.count()} chunks)")
            return current
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    for start in range(0, len(chunks), BATCH_ADD):
        stop = start + BATCH_ADD
        batch = chunks[start:stop]
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "strategy": chunk["strategy"],
                }
                for chunk in batch
            ],
            embeddings=embeddings[start:stop].tolist(),
        )
    print(f"Indexed {collection.count()} chunks in Chroma collection '{collection_name}'")
    return collection


class ConversationalMemory:
    """One-turn retrieval memory.

    The vector index is not updated. If the new question contains a pronoun
    or demonstrative, the previous user question is prepended so the
    embedding and the reranker both see the missing referent.
    """

    def __init__(self) -> None:
        self.history: list[dict] = []

    def rewrite(self, query: str) -> str:
        if not self.history or not ANAPHORA.search(query):
            return query
        previous = self.history[-1]["user"]
        return f"{previous} Follow-up question: {query}"

    def add(self, user: str, rewritten: str) -> None:
        self.history.append({"user": user, "rewritten": rewritten})


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for chunk_id in top if chunk_id in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for chunk_id in top if chunk_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def doc_ids_of(ranked: list[dict]) -> list[int]:
    return [item["doc_id"] for item in ranked]


def precision_doc_at_k(ranked_docs: list[int], gold_docs: set[int], k: int) -> float:
    top = ranked_docs[:k]
    if not top:
        return 0.0
    return sum(1 for doc_id in top if doc_id in gold_docs) / k


def recall_doc_at_k(ranked_docs: list[int], gold_docs: set[int], k: int) -> float:
    if not gold_docs:
        return 0.0
    found = {doc_id for doc_id in ranked_docs[:k] if doc_id in gold_docs}
    return len(found) / len(gold_docs)


def mrr_doc(ranked_docs: list[int], gold_docs: set[int]) -> float:
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in gold_docs:
            return 1.0 / rank
    return 0.0


def vector_search(collection: chromadb.Collection, query_embedding: np.ndarray, pool: int) -> list[dict]:
    count = collection.count()
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(pool, count),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for chunk_id, document, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append(
            {
                "id": chunk_id,
                "doc_id": int(metadata["doc_id"]),
                "chunk_index": int(metadata["chunk_index"]),
                "text": document,
                "vector_distance": float(distance),
            }
        )
    hits.sort(key=lambda item: item["vector_distance"])
    return hits


def rerank(reranker: CrossEncoder, query: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    pairs = [(query, hit["text"]) for hit in hits]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    ordered = []
    for hit, score in zip(hits, scores):
        updated = dict(hit)
        updated["rerank_score"] = float(score)
        ordered.append(updated)
    ordered.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ordered


def score_ranking(ranked: list[dict], relevant_ids: set[str], gold_docs: list[int]) -> dict:
    ids = [item["id"] for item in ranked]
    docs = doc_ids_of(ranked)
    gold = set(gold_docs)
    return {
        "precision_at_5": precision_at_k(ids, relevant_ids, TOP_K),
        "recall_at_5": recall_at_k(ids, relevant_ids, TOP_K),
        "mrr": mean_reciprocal_rank(ids, relevant_ids),
        "doc_precision_at_5": precision_doc_at_k(docs, gold, TOP_K),
        "doc_recall_at_5": recall_doc_at_k(docs, gold, TOP_K),
        "doc_mrr": mrr_doc(docs, gold),
        "n_relevant_in_index": len(relevant_ids),
    }


def preview(ranked: list[dict], relevant_ids: set[str], n: int = TOP_K) -> list[dict]:
    rows = []
    for rank, item in enumerate(ranked[:n], start=1):
        rows.append(
            {
                "rank": rank,
                "id": item["id"],
                "doc_id": item["doc_id"],
                "relevant": item["id"] in relevant_ids,
                "vector_distance": round(item.get("vector_distance", 0.0), 4),
                "rerank_score": None if "rerank_score" not in item else round(item["rerank_score"], 4),
                "snippet": item["text"][:280],
            }
        )
    return rows


def macro_average(metric_rows: list[dict]) -> dict:
    keys = [
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "doc_precision_at_5",
        "doc_recall_at_5",
        "doc_mrr",
    ]
    return {key: sum(row[key] for row in metric_rows) / len(metric_rows) for key in keys}


def print_metric_table(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n{title}")
    print(f"{'setting':<28} {'P@5':>8} {'R@5':>8} {'MRR':>8} {'docP@5':>8} {'docR@5':>8} {'docMRR':>8}")
    for name, metrics in rows:
        print(
            f"{name:<28} "
            f"{metrics['precision_at_5']:8.3f} "
            f"{metrics['recall_at_5']:8.3f} "
            f"{metrics['mrr']:8.3f} "
            f"{metrics['doc_precision_at_5']:8.3f} "
            f"{metrics['doc_recall_at_5']:8.3f} "
            f"{metrics['doc_mrr']:8.3f}"
        )


def print_hits(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    for row in rows:
        flag = "RELEVANT" if row["relevant"] else "not relevant"
        score = ""
        if row["rerank_score"] is not None:
            score = f" rerank={row['rerank_score']}"
        print(
            f"  {row['rank']}. doc {row['doc_id']} ({flag}) "
            f"distance={row['vector_distance']}{score}"
        )
        print(f"     {row['snippet']}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_documents(N_DOCS)
    print(f"Loaded {len(docs)} documents. Mean article length: {sum(len(d['article']) for d in docs) / len(docs):.0f} chars.")

    strategies = {
        "fixed_window": build_chunks(docs, "fixed_window"),
        "sentence_packed": build_chunks(docs, "sentence_packed"),
    }
    stats = {name: chunk_stats(chunks) for name, chunks in strategies.items()}
    print("Chunk stats:", json.dumps(stats))

    judgments = {}
    for strategy, chunks in strategies.items():
        judgments[strategy] = {}
        for spec in QUERIES + MEMORY_TURNS:
            ids = sorted(relevant_id_set(chunks, spec))
            judgments[strategy][spec["id"]] = ids
            print(f"  relevant chunks | {strategy:<16} {spec['id']:<22} {len(ids)}")
            if not ids:
                raise RuntimeError(
                    f"No relevant chunks for {spec['id']} under {strategy}. "
                    "The keyword rubric missed this paper; adjust the label before indexing."
                )

    (RESULTS_DIR / "relevance_judgments.json").write_text(
        json.dumps(
            {
                "rubric": (
                    "A chunk is relevant if its document was hand-labeled as the paper "
                    "that answers the query and the chunk contains the required phrases."
                ),
                "queries": [{k: v for k, v in spec.items() if k != "text"} | {"text": spec["text"]} for spec in QUERIES],
                "memory_turns": MEMORY_TURNS,
                "chunk_ids": judgments,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = {
        name: embed_chunks(embedder, chunks, ROOT / "data" / f"emb_{name}.npy")
        for name, chunks in strategies.items()
    }
    collections = {
        name: index_chunks(f"arxiv_{name}", chunks, embeddings[name])
        for name, chunks in strategies.items()
    }
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    def run_query(strategy: str, spec: dict, query_text: str) -> dict:
        relevant = set(judgments[strategy][spec["id"]])
        query_vec = embedder.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vector_hits = vector_search(collections[strategy], query_vec, CANDIDATE_POOL)
        reranked_hits = rerank(reranker, query_text, vector_hits)
        return {
            "query": query_text,
            "vector": {
                "metrics": score_ranking(vector_hits, relevant, spec["gold_docs"]),
                "top": preview(vector_hits, relevant),
            },
            "reranked": {
                "metrics": score_ranking(reranked_hits, relevant, spec["gold_docs"]),
                "top": preview(reranked_hits, relevant),
            },
        }

    per_query = {}
    for spec in QUERIES:
        print("\n" + "=" * 72)
        print(spec["id"])
        print(spec["text"])
        per_query[spec["id"]] = {}
        for strategy in strategies:
            outcome = run_query(strategy, spec, spec["text"])
            per_query[spec["id"]][strategy] = outcome
            print_hits(f"{strategy} | vector search", outcome["vector"]["top"])
            print_hits(f"{strategy} | after reranking", outcome["reranked"]["top"])

    comparison_rows = []
    averages = {}
    for strategy in strategies:
        for stage in ("vector", "reranked"):
            rows = [per_query[spec["id"]][strategy][stage]["metrics"] for spec in QUERIES]
            key = f"{strategy}/{stage}"
            averages[key] = macro_average(rows)
            comparison_rows.append((key, averages[key]))
    print_metric_table(f"Macro average over the {len(QUERIES)} standalone queries", comparison_rows)

    # Conversational memory. The index does not change. Only the follow-up
    # string changes, and only when rewrite() sees anaphora.
    memory = ConversationalMemory()
    turn1, turn2 = MEMORY_TURNS
    memory_report = {"turns": [], "followup_comparison": {}}

    print("\n" + "=" * 72)
    print("CONVERSATIONAL MEMORY")
    rewritten_1 = memory.rewrite(turn1["text"])
    print(f"\nTurn 1 user: {turn1['text']}")
    print(f"Turn 1 retrieval query: {rewritten_1}")
    turn1_runs = {strategy: run_query(strategy, turn1, rewritten_1) for strategy in strategies}
    memory.add(turn1["text"], rewritten_1)
    for strategy in strategies:
        print_hits(f"Turn 1 {strategy} | vector", turn1_runs[strategy]["vector"]["top"])

    raw_followup = turn2["text"]
    resolved_followup = memory.rewrite(raw_followup)
    print(f"\nTurn 2 user: {raw_followup}")
    print("Turn 2 WITHOUT memory:", raw_followup)
    print("Turn 2 WITH memory:   ", resolved_followup)

    followup = {}
    for strategy in strategies:
        without = run_query(strategy, turn2, raw_followup)
        with_memory = run_query(strategy, turn2, resolved_followup)
        followup[strategy] = {"without_memory": without, "with_memory": with_memory}
        print_hits(f"Turn 2 {strategy} | no memory | vector", without["vector"]["top"])
        print_hits(f"Turn 2 {strategy} | with memory | vector", with_memory["vector"]["top"])
        print_hits(f"Turn 2 {strategy} | no memory | reranked", without["reranked"]["top"])
        print_hits(f"Turn 2 {strategy} | with memory | reranked", with_memory["reranked"]["top"])

    memory_rows = []
    memory_averages = {}
    for strategy in strategies:
        for condition in ("without_memory", "with_memory"):
            for stage in ("vector", "reranked"):
                metrics = followup[strategy][condition][stage]["metrics"]
                key = f"{strategy}/{condition}/{stage}"
                memory_averages[key] = metrics
                memory_rows.append((key, metrics))
    print_metric_table("Follow-up query only (turn 2). Gold is the MTJ synapse paper.", memory_rows)

    memory_report = {
        "turn1_query": turn1["text"],
        "turn2_user": raw_followup,
        "turn2_rewritten": resolved_followup,
        "mechanism": (
            "If the new question contains a pronoun or demonstrative, prepend the "
            "previous user question before embedding and reranking."
        ),
        "turn1": turn1_runs,
        "turn2": followup,
        "metrics": memory_averages,
    }

    payload = {
        "corpus": {
            "dataset": "ccdv/arxiv-summarization",
            "config": "document",
            "split": "train[:100]",
            "n_docs": len(docs),
            "embedding_model": EMBED_MODEL_NAME,
            "reranker": RERANK_MODEL_NAME,
            "vector_db": f"chromadb persistent at {CHROMA_PATH}",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "candidate_pool": CANDIDATE_POOL,
            "k": TOP_K,
            "chunk_stats": stats,
        },
        "averages": averages,
        "per_query": per_query,
        "memory": memory_report,
    }
    out_path = RESULTS_DIR / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
