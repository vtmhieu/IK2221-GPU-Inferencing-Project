# Task 2: Scheduler for Batched Requests

In this task, you will implement a more complex scheduler to handle batched requests efficiently. The goal is to maximize cache reuse and improve the overall throughput and latency.

## Required Implementation
You need to implement the following blocks:

1.  **Batch Request Generator:** An extension of the request generator from Task 1. This new version must provide batches of requests instead of a single request at a time to the scheduler.
2.  **Batch Scheduler:** A scheduler that handles the batched requests and orders them so that requests which share the same context are processed as sequentially as possible.
    *   *Note: When you need to implement batched requests and reordering, you will most likely have to update the function `create_chat_completion` located in: `lmcache-vllm-extended/lmcache_vllm/custom_api.py`.*

## Performance Analysis
Based on the metrics defined in Task 1 (throughput and latency), analyze the performance of your new system **when you vary the amount of local cache**.

## Questions to Answer
1.  Does your scheduler improve the metrics you defined? Why?
2.  What happens if the local cache has a very high size? Does your scheduler have any impact in that case? How does the batch size of the grouped requests affect such impact?
3.  What happens when the diversity of the requests increases?
4.  What happens when requests have larger contexts sizes?
