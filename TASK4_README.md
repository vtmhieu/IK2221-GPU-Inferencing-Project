# Task 4 (Bonus): Clustering Requests Based on KV Cache Representations

**Points:** Up to 10 bonus points (can compensate for points lost in Tasks 1–3 or Assignment 1).

In the previous task (RAG), the goal was to retrieve the correct document context from the request prompt using semantic similarity in embedding space. In this task, we investigate whether it is possible to identify the underlying topic or source document of a request *after inference*, using only the cached Key-Value (KV) tensors stored in the LMCache engine.

If similar requests produce similar internal representations, we can use those to cluster and group semantically related requests, enabling downstream optimizations like intelligent cache reuse or eviction prioritization.

## Required Implementation
You must perform the following steps:

1.  **Extract KV Tensors:** Extract the KV cache tensors for a set of processed requests. Ensure that the original request topic (e.g., paper ID) is known for evaluation purposes.
2.  **Vector Representation Aggregation:** For each request, compute a fixed-size vector representation by aggregating KV values. For instance, you could average across attention heads and/or layers, or apply layer weighting.
3.  **Dimensionality Reduction:** Apply a dimensionality reduction method (e.g., PCA, Truncated SVD) to obtain a compact feature vector for each request. *Note: do not limit the reduced space to 2D or 3D; select a dimensionality that retains meaningful structure.*
4.  **Clustering:** Perform clustering (e.g., KMeans, DBSCAN) on the reduced KV cache representations.
5.  **Evaluation:** Evaluate the quality of the clustering by comparing predicted cluster labels with the known request topics from the request generator or RAG labels. (It is recommended to visualize the results in a 2D space, e.g. using PCA to 2D for plotting).

## Questions to Answer
1.  Are requests referring to the same document or topic clustered together based solely on their KV cache?
2.  What is the performance of your KV-clustering method compared to the previous RAG implementation?
3.  How could this representation-based clustering be used to improve request scheduling or prefetching mechanisms in LLM inference pipelines?
