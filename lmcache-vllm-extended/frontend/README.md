# Frontend & Benchmark — Tasks 1 and 2

This directory contains the Streamlit chat frontend, CLI client, and the benchmark tooling for Task 1 (Baseline Evaluation) and Task 2 (Batched Request Scheduling).

## Directory Structure

```bash
frontend/
├── chat_session.py        # OpenAI-compatible client wrapper (provided)
├── cli.py                 # Interactive CLI chat client (provided)
├── frontend.py            # Streamlit web UI (provided)
├── request_generator.py   # Generates benchmark requests from context files
├── batch_scheduler.py     # Task 2 scheduler for batched requests
├── benchmark.py           # Runs requests, measures latency & throughput
├── data/                  # Context .txt files (course paper summaries)
└── results/               # Benchmark output CSVs (auto-created)
```

## Prerequisites

Before running anything, make sure the following two services are running (in separate terminals):

### Terminal 1 — LMCache Server

```bash
cd ~/IK2221-GPU-Inferencing-Project
source ./venv/bin/activate
python3 -m lmcache_server.server 127.0.0.1 65432 /tmp/lmcache_storage
```

### Terminal 2 — vLLM Server

```bash
cd ~/IK2221-GPU-Inferencing-Project
source ./venv/bin/activate



# Then start:
LMCACHE_CONFIG_FILE=lmcache-vllm-extended/configuration.yaml CUDA_VISIBLE_DEVICES=0 python lmcache-vllm-extended/lmcache_vllm/script.py serve Qwen/Qwen2.5-1.5B-Instruct  --gpu-memory-utilization 0.8 --dtype half --port 8000 --guided-decoding-backend lm-format-enforcer
```

Wait until you see `Uvicorn running on http://0.0.0.0:8000` before proceeding.

---

## Running the Streamlit Frontend (should run in VScode if you are running on school server)

### Terminal 3 — Frontend

```bash
cd ~/IK2221-GPU-Inferencing-Project/
source ./venv/bin/activate
cd lmcache-vllm-extended/frontend
streamlit run frontend.py
```

Access via JupyterHub proxy: `https://<jupyterhub-url>/user/<username>/proxy/8501/`
Or just simply click on the url <http://localhost:8501/> provided after running the command above.

## Running the CLI Client

```bash
cd ~/IK2221-GPU-Inferencing-Project/lmcache-vllm-extended/frontend
python cli.py --data-dir data/
```

---

## Running Benchmarks (Task 1)

Open a **forth terminal** while both services are running:

```bash
cd ~/IK2221-GPU-Inferencing-Project/
source ./venv/bin/activate
cd lmcache-vllm-extended/frontend
```

### Request Generator — Preview Requests (Optional)

Preview generated requests without sending them to the server:

```bash
# Random mode — shuffled contexts
python request_generator.py --data-dir data/ --num-requests 20 --mode random

# Sequential mode — grouped by context
python request_generator.py --data-dir data/ --mode sequential

# Repeated mode — same context
python request_generator.py --data-dir data/ --mode repeated --context-id vllm --num-requests 10
```

### Q1 — Latency vs Sequence Length

Measures how latency changes as the combined context + question length increases.

```bash
python benchmark.py --mode increasing_length -o results/q1_seqlen.csv
```

The graph result is saved as a csv if we specify the output with -g

### Q2 — KV Cache Reuse (Re-feeding Old Requests)

Measures latency improvement when a previously-seen context is fed again.

```bash
python benchmark.py --mode repeated --context-id vllm --num-requests 10 -o results/q2_cache_reuse.csv
```

### Q3 — Request Diversity Impact

Measures performance when request diversity (number of contexts, randomness) increases.

```bash
python benchmark.py --mode random --num-requests 30 -o results/q3_diversity.csv
```

### Custom Output Path

```bash
python benchmark.py --mode random --num-requests 50 -o results/my_experiment.csv
```

## Running Benchmarks (Task 2)

Task 2 batching is opt-in. If you do not pass `--batch-size` or `--scheduler`, the benchmark keeps the original Task 1 single-request behavior and still sends normal OpenAI-compatible `/v2/chat/completions` requests.

The Task 2 scheduler is implemented as a separate reusable block in `batch_scheduler.py`. The benchmark first generates the same flat request list as Task 1, splits it into batches, optionally reorders each batch, and then sends requests through the existing `ChatSession.chat()` path.

### Baseline Batch Order

Runs batched input without reordering. Use this as the Task 2 baseline.

```bash
python benchmark.py --mode random --num-requests 40 \
  --batch-size 8 --scheduler none \
  -o results/task2_batch8_none.csv
```

### Grouped Scheduler

Groups requests within each batch by `context_id`, so requests that share the same context are served sequentially where possible.

```bash
python benchmark.py --mode random --num-requests 40 \
  --batch-size 8 --scheduler grouped \
  -o results/task2_batch8_grouped.csv
```

### Task 2 Experiment Notes

For fair baseline-vs-scheduler comparisons:

- Use the same `--seed`, `--mode`, `--num-requests`, and `--batch-size` for paired runs.
- Restart or clear LMCache/vLLM between paired runs, or document a fixed warmup procedure.
- Sweep local cache size by restarting vLLM with different `--gpu-memory-utilization` values.
- Sweep batch size, for example `2`, `4`, `8`, and `16`.
- Increase request diversity with `--mode random-extended` or more contexts.
- Use longer contexts or longer question templates when answering the larger-context-size question.

### Q1 - Does the scheduler improve the metrics?

Run the same request set twice: once without reordering and once with grouping.

```bash
python benchmark.py --mode random --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler none \
  -o results/task2_q1_none.csv

python benchmark.py --mode random --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler grouped \
  -o results/task2_q1_grouped.csv
```

Compare average `ttft_s`, average `total_latency_s`, and throughput. Also inspect `context_id`, `original_position`, and `scheduled_position` to confirm that the grouped run placed same-context requests next to each other within batches.

### Q2 - What happens with a very large local cache, and how does batch size matter?

Repeat the Q1 pair after restarting vLLM with a larger `--gpu-memory-utilization`. Keep the same seed and request count.

```bash
python benchmark.py --mode random --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler none \
  -o results/task2_q2_large_cache_none.csv

python benchmark.py --mode random --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler grouped \
  -o results/task2_q2_large_cache_grouped.csv
```

Then sweep batch size while keeping the cache size fixed:

```bash
for BATCH in 2 4 8 16; do
  python benchmark.py --mode random --seed 42 --num-requests 40 \
    --batch-size ${BATCH} --scheduler none \
    -o results/task2_q2_batch_${BATCH}_none.csv

  python benchmark.py --mode random --seed 42 --num-requests 40 \
    --batch-size ${BATCH} --scheduler grouped \
    -o results/task2_q2_batch_${BATCH}_grouped.csv
done
```

Record whether the scheduler benefit shrinks when the cache is large. For batch size, check whether larger batches give the scheduler more opportunities to group repeated `context_id` values.

### Q3 - What happens when request diversity increases?

Compare normal random requests with a more diverse request set. Run both scheduler modes for each workload.

```bash
python benchmark.py --mode random --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler none \
  -o results/task2_q3_random_none.csv

python benchmark.py --mode random --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler grouped \
  -o results/task2_q3_random_grouped.csv

python benchmark.py --mode random-extended --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler none \
  -o results/task2_q3_random_extended_none.csv

python benchmark.py --mode random-extended --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler grouped \
  -o results/task2_q3_random_extended_grouped.csv
```

For the report, compare the number of repeated `context_id` values inside each batch and the latency/throughput results. Higher diversity should usually reduce the benefit of grouping because there are fewer repeated contexts to reuse.

### Q4 - What happens when requests have larger context sizes?

Use longer context files if available, or use the most token-heavy requests in the generated CSVs. Keep the same scheduler comparison.

```bash
python benchmark.py --mode random-extended --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler none \
  -o results/task2_q4_large_context_none.csv

python benchmark.py --mode random-extended --seed 42 --num-requests 40 \
  --batch-size 8 --scheduler grouped \
  -o results/task2_q4_large_context_grouped.csv
```

In the CSVs, use `context_tokens` and `token_length` to separate shorter and longer requests. Report whether grouped scheduling helps more for longer contexts, since reusing a large context can avoid more repeated KV-cache work.

### Full CLI Options

```bash
python benchmark.py --help
```

| Flag                | Default     | Description                                  |
| ------------------- | ----------- | -------------------------------------------- |
| `--ip`              | `127.0.0.1` | vLLM server IP address                       |
| `--port`            | `8000`      | vLLM server port                             |
| `--data-dir`        | `data/`     | Folder containing context `.txt` files       |
| `--seed`            | `42`        | Random seed for reproducibility              |
| `--mode`            | `random`    | Request generation mode                      |
| `--num-requests`    | `30`        | Number of requests (random/repeated modes)   |
| `--num-per-context` | `3`         | Requests per context (sequential mode)       |
| `--batch-size`      | `1`         | Task 2 batch size; `1` keeps Task 1 behavior |
| `--scheduler`       | `none`      | Task 2 scheduler: `none` or `grouped`        |
| `--context-id`      | _(first)_   | Specific context for repeated mode           |
| `-o`, `--output`    | auto        | Output CSV path                              |
| `-g`,               | auto        | Output graph path                            |

---

## Sweeping Cache Sizes

To vary the allocated local cache memory (as required by Task 1), restart the vLLM server with different `--gpu-memory-utilization` values and re-run the benchmark:

```bash
# Example: run benchmark at 3 different cache sizes
for UTIL in 0.3 0.5 0.7; do
  # Restart vLLM with new utilization (in Terminal 2)
  # Then run:
  python benchmark.py --mode random --num-requests 30 \
    -o results/random_util_${UTIL}.csv
done
```

## Output Format

Results are saved as CSV with the following columns:

| Column               | Description                                |
| -------------------- | ------------------------------------------ |
| `request_id`         | Sequential request number                  |
| `context_id`         | Which context file was used                |
| `question`           | The question asked                         |
| `token_length`       | Context + question token count             |
| `context_tokens`     | Context-only token count                   |
| `ttft_s`             | Time to first token (seconds)              |
| `total_latency_s`    | Total response time (seconds)              |
| `response_length`    | Response character count                   |
| `response_preview`   | First 120 chars of model response          |
| `batch_id`           | Task 2 batch index, blank for Task 1 path  |
| `batch_size`         | Number of requests in the batch            |
| `original_position`  | Request position inside the incoming batch |
| `scheduled_position` | Request position after scheduling          |
| `scheduler_strategy` | Task 2 scheduler strategy used             |
