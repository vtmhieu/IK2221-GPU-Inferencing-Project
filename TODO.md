# Project TODO

## Setup (Prerequisite)

- [x] Obtain course paper summaries `.txt` files from Canvas and place them in `lmcache-vllm-extended/frontend/data/`
- [ ] Verify the vLLM + LMCache server starts correctly and the Streamlit frontend chat works end-to-end

---

## Task 1 — Baseline Evaluation

### Implementation

- [ ] `request_generator.py` — load all context `.txt` files, generate a stream of `(context_id, context_text, question)` tuples in randomized order; track which request maps to which context ID
- [ ] `benchmark.py` — send requests sequentially to `/v2/chat/completions`, record latency (time-to-first-token or total response time) and compute throughput (requests/sec)

### Experiments

- [ ] Sweep local cache size (`gpu_memory_utilization` or LMCache limit) across 3–5 values and record throughput + latency
- [ ] Collect data for the three analyses below

### Questions to Answer

- [ ] Q1: Plot sequence length (context + question tokens) vs response time
- [ ] Q2: Plot repeated-request latency vs cache size — does KV cache hit improve latency?
- [ ] Q3: Plot throughput/latency vs request diversity (more contexts, more random order)
- [ ] Choose optimal cache size to carry forward into Task 2

---

## Task 2 — Batch Scheduler

### Implementation

- [ ] `batch_request_generator.py` — extend Task 1 generator to emit batches of N requests at a time
- [ ] Batch scheduler in `lmcache-vllm-extended/lmcache_vllm/custom_api.py` (`create_chat_completion`) — reorder batch so requests sharing the same context are grouped sequentially to maximize KV cache reuse

### Experiments

- [ ] Repeat Task 1 benchmarks with the batch scheduler; sweep cache size and batch size
- [ ] Compare throughput and latency against Task 1 baseline

### Questions to Answer

- [ ] Q1: Does the scheduler improve throughput/latency? Why?
- [ ] Q2: What happens with a very large cache? Does batch size matter in that case?
- [ ] Q3: What happens when request diversity increases?
- [ ] Q4: What happens with larger context sizes?

---

## Task 3 — RAG Pipeline

### Implementation

- [ ] `rag_request_generator.py` — generate questions from summarized paper content without revealing the context/doc ID in the request
- [ ] `rag_database.py` — load summarized papers, compute embeddings with a HuggingFace model (e.g. `Qwen/Qwen2.5-1.5B-Instruct`), store `{doc_id: embedding}` mapping
- [ ] `rag_retriever.py` — cosine similarity search: given a question embedding, return the most similar doc and its full context text
- [ ] Scheduler integration — retrieve context via RAG, group requests by retrieved doc ID, send batches in sequential order (reuse Task 2 scheduler)

### Experiments

- [ ] Measure RAG retrieval accuracy: compare retrieved doc ID vs ground-truth ID from the request generator
- [ ] Measure latency breakdown: retrieval time vs inference time
- [ ] Sweep number of docs in RAG DB and record accuracy and response time

### Questions to Answer

- [ ] Q1: What is the RAG retrieval accuracy? How does it vary with DB size?
- [ ] Q2: How does the RAG pipeline determine the most relevant document?
- [ ] Q3: How does retrieval time affect overall inference latency?
- [ ] Q4: How does increasing the number of documents affect accuracy and response time?

---

## Task 4 — KV Cache Clustering (Bonus, +10 pts)

### Implementation

- [ ] Extract KV tensors per request from LMCache; keep ground-truth doc ID for evaluation
- [ ] `kv_aggregator.py` — average KV tensors across attention heads and layers → fixed-size vector per request
- [ ] Apply PCA or Truncated SVD; choose `n_components` to retain ~90% variance (do not limit to 2D/3D)
- [ ] Run KMeans or DBSCAN on the reduced vectors
- [ ] Evaluate: compare predicted cluster labels vs ground-truth doc IDs (NMI or accuracy)
- [ ] Visualize clusters in 2D (PCA to 2D, scatter plot colored by true doc ID)

### Questions to Answer

- [ ] Q1: Are requests from the same document clustered together using only KV cache?
- [ ] Q2: How does KV-clustering accuracy compare to the RAG retrieval accuracy?
- [ ] Q3: How could KV-based clustering improve request scheduling or cache prefetching?

---

## Report

- [ ] Combine all plots and answers into the final project report
- [ ] Document the optimal cache size chosen at the end of Task 1 and justify its use in Tasks 2–3
