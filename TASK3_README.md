# Task 3: RAG (Retrieval-Augmented Generation)

RAG is a powerful tool to store embeddings and retrieve corresponding context based on similarities between query inputs and stored embeddings. In this task, you will integrate RAG into your system pipeline.

Unlike the previous contexts, you cannot assume that the request packets include the ID of the document to be used as context for the LLM inferencing. Instead, the context must be retrieved automatically.

## Required Implementation
You will generate embeddings using any model (e.g., Llama or Qwen) from the Hugging Face library (via PyTorch) and compute semantic similarities (e.g., cosine similarity). 

The additional architecture blocks will consist of:
1.  **Request Generator:** A generator based on the available summarized papers provided as input in the resources.
2.  **RAG Database:** A database that maps summarized papers to their embeddings, computed with a Hugging Face transformer. As the original text for extracting embeddings, use summarized versions of the papers reviewed during the course.
3.  **Search-and-Retrieve Mechanism:** A RAG search mechanism that provides the most similar paper given a request, based on computed embeddings using cosine similarity or another distance-based metric.
4.  **Scheduler Integration:** A scheduler that orders the classified requests from the RAG module in a way that requests with the same contexts are grouped sequentially to minimize redundant KV cache loads.

### Example Embedding Generation Code
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_llm_embeddings(text, model, tokenizer):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        # Forward pass through the model to get hidden states
        outputs = model(**inputs, output_hidden_states=True)
        
        # Extract the last hidden state (embeddings)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        
        # Mean pooling for sentence-level embedding
        embeddings = last_hidden_state.mean(dim=1)
        
    return embeddings
```

## Performance Analysis & Questions to Answer
Analyze the performance of your RAG model (especially when varying the local cache size) and answer the following questions:

1.  What is the accuracy of the RAG module in detecting the correct context for the requests? (Employ the information from the request generator to address this question).
2.  How does the RAG pipeline determine which document is most relevant to a given prompt?
3.  How does context retrieval time impact the overall inference latency?
4.  How does increasing the number of documents in the RAG database affect accuracy and response time?
