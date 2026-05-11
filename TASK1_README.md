# Task 1: Evaluation of LLM Inference (With and Without KV Caching)

In this task, you will run and analyze the behavior of the base system implementation using a simple, unoptimized scheduler. The baseline scheduler simply forwards incoming requests to the model without any reordering.

## Getting Started / Setup
1. A simple scheduler is already implemented for you, available at:
   `https://github.com/ali-bana/lmcache-vllm-extended/blob/ik2221/lmcache_vllm/custom_api.py`
2. Install the required libraries as defined in the project repository:
   `https://github.com/ali-bana/lmcache-vllm-extended/tree/ik2221`
3. Ensure that inference from the frontend interface is working correctly before proceeding to implement the request generator and perform your analysis.

## Required Implementation
You are required to implement the following component:
*   **Request Generator:** A component that creates a set of requests over multiple contexts provided in the resources. The generator should create a stream of unordered, different requests that are handled sequentially by the LLM. *Make sure that you know in advance which request corresponds to which context.*
*   **Resources:** Use the collection of `.txt` files related to the course papers found on Canvas. These documents act as baseline contexts for your analysis. You can also extend this set with additional contexts for your experiments.

## Performance Analysis
Once your request generator is ready, analyze the performance of the inference pipeline by measuring:
1.  **Throughput:** The number of handled requests per second.
2.  **Latency:** The request response time.

You must measure these metrics **while varying the amount of allocated memory** in the local cache of the LLM model.

## Questions to Answer
1.  What happens to the latency as the combined context + question length increases? (Plot a graph showing the relationship between sequence length and response time).
2.  What happens if an old (previously seen) request is fed again to the model? Are the performances better in terms of latency? Analyze how the size of the local cache affects these results.
3.  What happens to the performance when the diversity (e.g., increasing the number of contexts, making the order of the requests more random) of the requests increases?

**Next Steps:** After completing these experiments and answering the questions, determine the optimal amount of allocated memory needed to perform inference with low latency and high throughput. Configure your system accordingly before proceeding to Task 2.

BABA
