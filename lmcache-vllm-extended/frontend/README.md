# Frontend & Benchmark — Task 1

This directory contains the Streamlit chat frontend, CLI client, and the benchmark tooling for Task 1 (Baseline Evaluation).

## Directory Structure

```bash
frontend/
├── chat_session.py        # OpenAI-compatible client wrapper (provided)
├── cli.py                 # Interactive CLI chat client (provided)
├── frontend.py            # Streamlit web UI (provided)
├── request_generator.py   # Generates benchmark requests from context files
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
python benchmark.py --mode sequential --num-per-context 3 -o results/q1_seqlen.csv
```

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

### Full CLI Options

```bash
python benchmark.py --help
```

| Flag               | Default     | Description                                      |
|--------------------|-------------|--------------------------------------------------|
| `--ip`             | `127.0.0.1` | vLLM server IP address                           |
| `--port`           | `8000`      | vLLM server port                                 |
| `--data-dir`       | `data/`     | Folder containing context `.txt` files           |
| `--seed`           | `42`        | Random seed for reproducibility                  |
| `--mode`           | `random`    | `random`, `sequential`, or `repeated`            |
| `--num-requests`   | `30`        | Number of requests (random/repeated modes)       |
| `--num-per-context`| `3`         | Requests per context (sequential mode)           |
| `--context-id`     | *(first)*   | Specific context for repeated mode               |
| `-o`, `--output`   | auto        | Output CSV path                                  |

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

| Column            | Description                                   |
|-------------------|-----------------------------------------------|
| `request_id`      | Sequential request number                     |
| `context_id`      | Which context file was used                   |
| `question`        | The question asked                            |
| `seq_length`      | Context + question character count            |
| `context_length`  | Context-only character count                  |
| `ttft_s`          | Time to first token (seconds)                 |
| `total_latency_s` | Total response time (seconds)                 |
| `response_length` | Response character count                      |
| `response_preview`| First 120 chars of model response             |
