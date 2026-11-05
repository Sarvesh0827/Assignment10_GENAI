# Improving RAG with Reranking, Chunking, and Conversational Memory

## Question

Does a second-stage reranker, a sentence-aware chunker, or a one-turn query memory retrieve more useful passages from full arXiv papers than a plain bi-encoder over fixed windows?

## Data

The corpus is the `document` configuration of [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization), train split, first 100 papers. These are full articles, not a pre-chunked release. The mean article length is 36,371 characters.

Preprocessing, applied before either chunker:

- `@xmath` tokens (the dataset's math placeholders) become `[MATH]`.
- `@xcite` tokens become `[CITE]`.
- Whitespace is collapsed to single spaces.

The abstract is kept for labeling but is not indexed. Only the article body is chunked. That choice matters for one query below: paper 7 spells out "magnetic tunnel junction" in the abstract and then almost only uses the acronym `mtj` in the body.

## Chunking

Both strategies use a budget of 800 characters. Chunks shorter than 180 characters are dropped.

| Strategy | Rule | Chunks | Mean length |
| --- | --- | ---: | ---: |
| `fixed_window` | Sliding window, 120-character overlap. A chunk can start or end mid-sentence. | 5,203 | 794 |
| `sentence_packed` | Sentences are packed until the budget is full. The last sentence is repeated on the next chunk when it still fits. A sentence longer than 800 characters is the only case that is cut. | 6,095 | 703 |

Sentence packing produces more chunks because of the one-sentence overlap and because chunks stop at a sentence boundary instead of filling every window to 800 characters.

## Index

Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalized). The two chunkings are stored as separate persistent Chroma collections, `arxiv_fixed_window` and `arxiv_sentence_packed`, under `chroma_db/`, with cosine distance. Chroma is local, so no hosted vector-database key is required. Pinecone or Redis would replace the `add` / `query` calls and leave the rest of the pipeline the same.

Retrieval asks Chroma for 20 candidates. The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. It scores each (query, chunk) pair and reorders those 20. It cannot recover a chunk the bi-encoder left outside the pool. Reported cutoffs use K = 5.

## What counts as relevant

Labels are manual. For each query, the abstracts of the 100 papers were read and one paper was marked as the source that answers it. A chunk from that paper is useful when it also contains the answer-bearing phrases listed in `QUERIES` inside `rag_assignment.py`. A chunk from the right paper that only contains background is not useful. The chunk ids that pass this rubric are saved in `results/relevance_judgments.json`.

There are six standalone queries:

| Id | Query | Paper | Useful chunks (fixed / sentence) |
| --- | --- | ---: | --- |
| q1 | What learning rates are obtained for regularized kernel methods applied to additive models? | 0 | 9 / 13 |
| q2 | How is a leptonic decay constant extracted from tagged meson decays? | 1 | 5 / 7 |
| q3 | How do nanoelectronic synapses store synaptic memory for neuromorphic computing? | 7 | 1 / 3 |
| q4 | How can origin-destination passenger counts be estimated for public transport? | 32 | 10 / 11 |
| q5 | Why does magnetoresistance stay linear instead of saturating on a topological insulator surface? | 20 | 4 / 7 |
| q6 | How are banks of waveform templates used to detect compact-object mergers in gravitational-wave data? | 27 | 22 / 30 |

q2 does not name CLEO-c, even though paper 1 is that experiment. q5 has nearby distractors: other papers mention magnetoresistance or topological insulators. q3 is strict on purpose. The body of paper 7 must contain both `mtj` and `plasticity` in the same chunk. Introductions about brain power fail that test.

Metrics:

- Precision@5: fraction of the top 5 chunks that are useful.
- Recall@5: fraction of all useful chunks in that index that appear in the top 5.
- MRR: reciprocal rank of the first useful chunk, measured on the ranked list of up to 20.
- Document-level versions of the same three scores treat a hit as "this chunk comes from the labeled paper."

Recall@5 is sensitive to how many chunks the rubric accepts. Sentence packing keeps answer sentences intact, so it often creates more useful chunks. Five hits out of 30 is a lower recall than five hits out of 22 even when the top of the list is equally good. Precision@5, MRR, and the raw count of useful chunks in the top 5 are the fairer chunking comparison. All three requested metrics are still reported.

## Sample retrievals

### Query: leptonic decay constant (q2)

Bi-encoder, sentence packing, rank 1 (useful):

> in these decays, strong interaction effects can be parametrized by a single quantity, [MATH], the pseudoscalar meson decay constant. the leptonic decay rate can be measured by experiment...

After reranking, rank 1 (useful, and closer to the actual measurement):

> in summary, using the sample of [MATH] tagged [MATH] decays with the cleo - c detector we obtain the absolute branching fraction of the leptonic decay...

Sentence-packed Precision@5 moves from 0.80 to 1.00. Fixed-window Precision@5 moves from 0.60 to 0.80. Every top-5 chunk is already from paper 1, so the reranker is reordering passages inside the right paper.

### Query: linear magnetoresistance (q5)

Fixed-window Precision@5 moves from 0.40 to 0.60. Sentence-packed Precision@5 moves from 0.80 to 1.00. The sentence chunker keeps the result sentence together:

> we find that a positive , nonsaturating and dominantly linear magnetoresistance can appear within quite wide magnetic - field range in the ti surface state...

Reranking then fills the rest of the top 5 with other passages from the same argument (Landau-level broadening and non-saturating magnetoresistance).

### Query where the metric hides a better top passage (q1)

On the fixed-window index, Precision@5 and MRR are already 1.00 before reranking. The rank-1 passage is still a generic opening ("additive models provide an important family of models..."). After reranking, rank 1 is the passage that states the result: learning rates of regularized kernel methods for additive models. The cutoff metrics do not move. The page a reader would actually see does.

## Conversational memory

Memory is a retrieval-time rewrite. The index is not updated. If the new question contains a pronoun or a demonstrative (`it`, `that`, `they`, ...), the previous user question is prepended and that expanded string is what gets embedded and reranked.

Turn 1: "How do magnetic tunnel junctions act as nanoelectronic synapses for neuromorphic memory?"

Turn 2, as typed: "How does the frequency of the input affect it?"

Turn 2, rewritten: "How do magnetic tunnel junctions act as nanoelectronic synapses for neuromorphic memory? Follow-up question: How does the frequency of the input affect it?"

The follow-up by itself is about frequency. That word shows up in lasers, cavities, tuners, and superconductors in this 100-paper sample. The intended answer is paper 7, which argues that stimulus frequency decides whether an MTJ synapse shows long-term potentiation.

Without memory, fixed-window vector search returns papers 53 and 33 in the top 5 (a tuner paper and an optics paper). The correct paper is not in the top 5 (document MRR 0.077, chunk Precision@5 0.00).

With memory, all five fixed-window hits are paper 7 (document Precision@5 1.00, document MRR 1.00). Reranking that memory-expanded query raises chunk Precision@5 from 0.40 to 0.60. The useful passages discuss stimulus frequency, short-term versus long-term plasticity, and nanoelectronic synapses.

Reranking the unresolved follow-up can still drag paper 7 up to rank 1 (document MRR goes to 1.00 on both chunkers), because one candidate in the pool of 20 happens to be the right paper and the cross-encoder prefers it. Chunk precision stays poor (0.00 fixed, 0.20 sentence): the rest of the list is still other papers talking about frequency. Memory is what makes the whole top 5 the right paper. Reranking then sorts passages inside it.

## Metrics

Macro average over the six standalone queries:

| Setting | P@5 | R@5 | MRR | Doc P@5 | Doc R@5 | Doc MRR | Useful chunks in top 5 (sum over 6 queries, out of 30) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed window, no rerank | 0.633 | 0.530 | 0.867 | 1.000 | 1.000 | 1.000 | 19 |
| Fixed window, reranked | 0.767 | 0.639 | 0.867 | 1.000 | 1.000 | 1.000 | 23 |
| Sentence packed, no rerank | 0.733 | 0.345 | 0.847 | 1.000 | 1.000 | 1.000 | 22 |
| Sentence packed, reranked | 0.833 | 0.406 | 0.842 | 1.000 | 1.000 | 1.000 | 25 |

Per-query Precision@5 (MRR in parentheses when it is below 1):

| Query | Fixed, vector | Fixed, rerank | Sentence, vector | Sentence, rerank |
| --- | ---: | ---: | ---: | ---: |
| q1 additive kernels | 1.00 | 1.00 | 0.80 | 1.00 |
| q2 leptonic decay | 0.60 | 0.80 | 0.80 | 1.00 |
| q3 synaptic memory | 0.20 (MRR 0.20) | 0.20 (MRR 0.20) | 0.00 (MRR 0.08) | 0.00 (MRR 0.05) |
| q4 OD matrix | 0.60 | 1.00 | 1.00 | 1.00 |
| q5 magnetoresistance | 0.40 | 0.60 | 0.80 | 1.00 |
| q6 template bank | 1.00 | 1.00 | 1.00 | 1.00 |

Follow-up only ("How does the frequency of the input affect it?"). Gold chunks are paper 7 passages that discuss frequency in the synaptic setting. There are 5 such chunks in each index.

| Setting | P@5 | R@5 | MRR | Doc P@5 | Doc MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed, no memory, vector | 0.00 | 0.00 | 0.00 | 0.00 | 0.077 |
| Fixed, no memory, reranked | 0.00 | 0.00 | 0.00 | 0.20 | 1.000 |
| Fixed, with memory, vector | 0.40 | 0.40 | 0.50 | 1.00 | 1.000 |
| Fixed, with memory, reranked | 0.60 | 0.60 | 0.50 | 1.00 | 1.000 |
| Sentence, no memory, vector | 0.00 | 0.00 | 0.05 | 0.00 | 0.050 |
| Sentence, no memory, reranked | 0.20 | 0.20 | 1.00 | 0.20 | 1.000 |
| Sentence, with memory, vector | 0.20 | 0.20 | 0.50 | 1.00 | 1.000 |
| Sentence, with memory, reranked | 0.40 | 0.40 | 1.00 | 1.00 | 1.000 |

## Comparison

### Without reranking vs with reranking

On standalone questions the bi-encoder already places the correct paper at rank 1. Document Precision@5, document Recall@5, and document MRR are 1.00 for every standalone setting. Reranking does not change which paper is found. It changes which passages from that paper sit in the top 5.

Across six queries, reranking adds 4 useful chunks to the fixed-window top-5 lists (19 to 23) and 3 to the sentence-packed lists (22 to 25). Macro Precision@5 rises by 0.134 (fixed) and 0.100 (sentence). Macro Recall@5 rises with them. The gains are concentrated on q2, q4, and q5, where background paragraphs from the right paper were occupying slots.

MRR barely moves, because the first useful chunk was usually already rank 1. The exception is q3 on the sentence index: reranking lowered MRR from 0.08 to 0.05. The cross-encoder preferred the neuromorphic introduction, and the few chunks that jointly mention `mtj` and `plasticity` sank to the bottom of the pool of 20. A reranker trained on MS MARCO web passages is not a specialist at this acronym-heavy paper, and it can only reorder what the first stage returned.

### Fixed windows vs sentence packing

Sentence packing has the higher Precision@5 in both stages (0.733 vs 0.633 without reranking, 0.833 vs 0.767 with reranking) and the higher count of useful top-5 chunks. q5 is the clearest case: keeping the magnetoresistance result inside one sentence raises Precision@5 from 0.40 to 0.80 before any reranking.

Fixed windows have the higher Recall@5 (0.530 vs 0.345, then 0.639 vs 0.406). Part of that gap is the denominator. The same rubric marks fewer fixed-window chunks as useful, because an 800-character cut often separates the two phrases that the label requires. q6 is the clean illustration: both methods place five useful chunks in the top 5, but recall is 5/22 for fixed windows and 5/30 for sentence packing.

q3 is the chunking failure. The fixed index has only one chunk that contains both `mtj` and `plasticity`, and it lands at rank 5. The sentence index has three, and none of them land in the top 5. The query says "nanoelectronic synapses" and "synaptic memory," which matches the introduction. The device name the paper actually uses in the body is `mtj`. Chunk boundaries cannot repair a query that never overlaps the wording of the answer span. Indexing the abstract, or expanding `mtj` during preprocessing, would.

### Without memory vs with memory

This is the largest gap in the experiment. The follow-up "How does the frequency of the input affect it?" has chunk Precision@5 of 0.00 on both vector indexes. Prepending the previous question about magnetic tunnel junctions raises document Precision@5 from 0.00 to 1.00 on both chunkers, and raises chunk Precision@5 to 0.40 (fixed) or 0.20 (sentence). Adding the reranker on top of memory reaches 0.60 and 0.40.

Reranking without memory is a partial substitute. It can promote the single correct candidate to rank 1, which is why document MRR becomes 1.00, but four of the five displayed chunks are still the wrong papers. Memory changes the query, so the candidate pool itself changes. That is a different operation from reordering a bad pool.

## Findings

1. On this 100-paper sample, topical retrieval of the right article is already solved by a small bi-encoder. Every standalone query has document MRR 1.00. The remaining errors are passage selection inside that article.
2. Cross-encoder reranking of the top 20 is a reliable passage selector here: Precision@5 goes from 0.633 to 0.767 (fixed windows) and from 0.733 to 0.833 (sentence packing). It does not help when the answer wording is an acronym the query never uses (q3).
3. Sentence packing beats fixed windows on Precision@5 and on the number of useful chunks shown to a reader. Fixed windows look better on Recall@5 mostly because they create fewer chunks that satisfy a phrase-conjunction label.
4. A one-line conversational memory, "prepend the previous question when the new one contains a pronoun," is what makes the follow-up retrievable. Reranking helps after that. It does not replace the missing referent.
5. The abstract is worth indexing. Paper 7's expanded device name occurs there once and never in the body. A query written the way a person talks ("magnetic tunnel junction") is then depending on the embedding model to bridge `mtj`.

## Limits

Six queries and one two-turn dialogue are enough to see the pattern and too few to estimate a stable margin. The usefulness labels are phrase checks on a hand-chosen paper, so a paraphrased answer can be marked irrelevant and a chunk that merely contains the phrases can be marked relevant. Recall@5 cannot exceed 5 / (number of useful chunks), and that number differs by chunker. The reranker never sees past rank 20. MiniLM truncates inputs around 256 wordpieces; 800 characters of this corpus usually fits, so truncation is not the main error. There is no generative answer step. The comparison is about what would be placed in the prompt.

## How to reproduce

```text
.venv\Scripts\python.exe rag_assignment.py
```

Dependencies are pinned in `requirements.txt`. The first run downloads the 100 papers, MiniLM, and the cross-encoder. Later runs reuse `data/arxiv_100.jsonl`, `data/emb_*.npy`, and `chroma_db/`. Numbers in this report are copied from `results/metrics.json`.
