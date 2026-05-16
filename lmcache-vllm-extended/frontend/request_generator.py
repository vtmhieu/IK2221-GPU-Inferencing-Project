"""
Request Generator for Task 1 — Baseline Evaluation

Loads context .txt files from a data directory and generates a randomized
stream of (context_id, context_text, question) tuples.  Each request is
tagged so the caller always knows which context it belongs to.

Usage:
    # As a module
    from request_generator import RequestGenerator
    gen = RequestGenerator("data/")
    for req in gen.generate(num_requests=50):
        print(req.context_id, req.question)

    # Standalone test
    python request_generator.py --data-dir data/ --num-requests 20
"""

import os
import random
from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer


# Pre-defined question templates — each is generic enough to apply to any
# research-paper context.  Extend this list to increase diversity.
QUESTION_TEMPLATES = [
    "What is the main contribution of this paper?",
    "What problem does this paper address?",
    "Summarize the key findings in 2-3 sentences.",
    "What methodology or approach is proposed?",
    "What are the main limitations mentioned by the authors?",
    "How does this work compare to prior approaches?",
    "What datasets or benchmarks were used for evaluation?",
    "What are the key performance metrics reported?",
    "What future work do the authors suggest?",
    "Explain the system architecture described in this paper.",
    "What is the motivation behind this research?",
    "What are the practical implications of this work?",
    "Describe the experimental setup used in this paper.",
    "What trade-offs does the proposed solution involve?",
    "How does this paper relate to network systems or ML inference?",
    "Who are the authors of this paper?",
    "Does the authors provide metrics about latency?"
]
QUESTION_TEMPLATES_MEDIUM = [
    """Read the paper carefully and explain the primary technical contribution. Include:
    1. The core problem being addressed
    2. The proposed solution
    3. Why the proposed approach is different from prior work
    4. The most important experimental result
    Keep the response under 200 words""",
    """Provide a structured summary of this paper. Your summary should include:
    - Research motivation
    - Problem statement
    - System architecture or methodology
    - Experimental setup
    - Main findings
    - Limitations
    - Suggested future work

    Use concise technical language and organize the response with bullet points.""",
    """Compare the approach proposed in this paper with previous approaches discussed by the authors. Focus specifically on:
    - Architectural differences
    - Performance trade-offs
    - Scalability considerations
    - Latency implications
    - Resource efficiency

    If numerical metrics are available, include them in the comparison.""",
    """ 
    Analyze the evaluation methodology used in this paper. Explain:
    - Which datasets or benchmarks were used
    - Why these benchmarks are appropriate
    - Which metrics were reported
    - Whether the experiments are sufficient to validate the claims
    - Any weaknesses or missing evaluations

    Conclude with your assessment of the rigor of the evaluation."""
]

QUESTION_TEMPLATES_LARGE = [
    """You are an expert researcher in distributed systems, machine learning systems, and large-scale inference infrastructure. Carefully read the provided paper and produce a comprehensive technical review intended for an audience of systems researchers and ML infrastructure engineers.

Your review should contain the following sections:

1. Executive Summary
Provide a concise overview of the paper’s goals, contributions, and findings.

2. Research Motivation
Explain the broader systems or machine learning problem being addressed. Discuss why this problem matters in practice, including scalability, latency, throughput, reliability, or efficiency concerns.

3. Technical Contributions
Describe the main innovations proposed by the paper. For each contribution:
- Explain the idea clearly
- Describe how it differs from prior work
- Explain why it is technically important

4. System Architecture and Design
Provide a detailed explanation of the architecture described in the paper. Include:
- Major system components
- Data flow
- Scheduling or orchestration mechanisms
- Memory management strategies
- Network communication patterns
- Inference or training pipeline details if applicable

5. Experimental Methodology
Describe the evaluation setup in detail:
- Hardware configuration
- Cluster or distributed setup
- Datasets and benchmarks
- Baselines used for comparison
- Metrics collected
- Ablation studies
- Reproducibility considerations

6. Performance Analysis
Analyze all reported performance metrics. Include discussion of:
- Latency
- Throughput
- Scalability
- Resource utilization
- Cost efficiency
- Accuracy or quality trade-offs
- Tail latency or percentile metrics if available

7. Comparison with Prior Work
Compare this paper against prior approaches mentioned in the related work section. Discuss:
- Advantages
- Limitations
- Novelty
- Engineering complexity
- Deployment practicality

8. Limitations and Trade-offs
Explain all limitations acknowledged by the authors, as well as any additional weaknesses you identify.

9. Future Research Directions
Summarize future work proposed by the authors and suggest additional research directions that could extend this work.

10. Final Assessment
Provide an overall technical assessment of the paper’s significance, strengths, weaknesses, and likely practical impact.

Your response should be highly detailed, technically rigorous, and approximately 1200-1500 words long."""
]


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Request:
    """A single benchmark request."""
    request_id: int
    context_id: str          # filename, e.g. "vllm"
    context_text: str        # content of the full document
    question: str            # question to ask about the context
    context_tokens: int = 0  # number of tokens in the context
    question_tokens: int = 0 # number of tokens in the question
    token_length: int = 0    # total tokens (context + question)


class RequestGenerator:
    """
    Generates benchmark requests from a folder of .txt context files.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing .txt context files.
    questions : list[str] | None
        Custom question list. Defaults to QUESTION_TEMPLATES.
    seed : int | None
        Random seed for reproducibility.  None = non-deterministic.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        extended_data_dir: str = "additionaldata/",
        questions: List[str] | None = None,
        seed: int | None = 42,
        model_name: str = DEFAULT_MODEL_NAME,
        tokenizer=None,
    ):
        self.data_dir = data_dir
        self.extended_data_dir = extended_data_dir
        self.questions = questions or QUESTION_TEMPLATES
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.contexts = self._load_contexts()
        self.extended_contexts = self._load_extended_contexts()

        if not self.contexts:
            raise FileNotFoundError(
                f"No .txt files found in '{data_dir}'. "
                "Make sure context files are present."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _load_contexts(self) -> dict:
        """Load all .txt files from data_dir into {context_id: text}."""
        contexts = {}
        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.endswith(".txt"):
                continue
            context_id = filename.removesuffix(".txt")
            with open(os.path.join(self.data_dir, filename), "r") as f:
                contexts[context_id] = f.read()
        return contexts

    def _load_extended_contexts(self) -> dict:
        """Load all .txt files from extended_data_dir into {context_id: text}."""
        contexts = {}
        for filename in sorted(os.listdir(self.extended_data_dir)):
            if not filename.endswith(".txt"):
                continue
            context_id = filename.removesuffix(".txt")
            with open(os.path.join(self.extended_data_dir, filename), "r") as f:
                contexts[context_id] = f.read()
        return contexts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_context_ids(self) -> List[str]:
        """Return the list of all loaded context IDs."""
        return list(self.contexts.keys())

    def get_extended_context_ids(self) -> List[str]:
        """Return the list of all loaded context IDs."""
        return list(self.extended_contexts.keys())

    def generate(self, num_requests: int = 50) -> List[Request]:
        """
        Generate a list of randomized requests.

        Each request pairs a randomly chosen context with a randomly chosen
        question.  The order is shuffled so that consecutive requests are
        unlikely to share the same context — this exercises the KV cache
        eviction behavior.
        """
        context_ids = list(self.contexts.keys())
        requests = []

        for i in range(num_requests):
            ctx_id = self.rng.choice(context_ids)
            question = self.rng.choice(self.questions)
            context_tokens = self._count_tokens(self.contexts[ctx_id])
            question_tokens = self._count_tokens(question)
            requests.append(
                Request(
                    request_id=i,
                    context_id=ctx_id,
                    context_text=self.contexts[ctx_id],
                    question=question,
                    context_tokens=context_tokens,
                    question_tokens=question_tokens,
                    token_length=context_tokens + question_tokens,
                )
            )

        # Shuffle to maximize context diversity between consecutive requests
        self.rng.shuffle(requests)
        return requests

    def generate_extended(self, num_requests: int = 50) -> List[Request]:
        """
        Generate a list of randomized requests.

        Each request pairs a randomly chosen extended context with a randomly chosen
        question.  The order is shuffled so that consecutive requests are
        unlikely to share the same context — this exercises the KV cache
        eviction behavior.
        """
        extended_context_ids = list(self.extended_contexts.keys())
        requests = []

        for i in range(num_requests):
            extended_ctx_id = self.rng.choice(extended_context_ids)
            question = self.rng.choice(self.questions)
            extended_context_tokens = self._count_tokens(self.extended_contexts[extended_ctx_id])
            question_tokens = self._count_tokens(question)
            requests.append(
                Request(
                    request_id=i,
                    context_id=extended_ctx_id,
                    context_text=self.extended_contexts[extended_ctx_id],
                    question=question,
                    context_tokens=extended_context_tokens,
                    question_tokens=question_tokens,
                    token_length=extended_context_tokens + question_tokens,
                )
            )

        # Shuffle to maximize context diversity between consecutive requests
        self.rng.shuffle(requests)
        return requests

    def generate_sequential(self, num_per_context: int = 3) -> List[Request]:
        """
        Generate requests grouped by context (not shuffled).

        Useful for measuring the benefit of KV cache hits when the same
        context is reused across consecutive requests.
        """
        requests = []
        req_id = 0
        for ctx_id, ctx_text in self.contexts.items():
            chosen_questions = self.rng.sample(
                self.questions,
                min(num_per_context, len(self.questions)),
            )
            for question in chosen_questions:
                context_tokens = self._count_tokens(ctx_text)
                question_tokens = self._count_tokens(question)
                requests.append(
                    Request(
                        request_id=req_id,
                        context_id=ctx_id,
                        context_text=ctx_text,
                        question=question,
                        context_tokens=context_tokens,
                        question_tokens=question_tokens,
                        token_length=context_tokens + question_tokens,
                    )
                )
                req_id += 1
        return requests

    def generate_increasing_length(self) -> List[Request]:
        """
        Generate 10 requests with increasing prompt complexity.

        Uses 6 short, 3 medium, and 1 large question templates, each
        paired with a distinct context.
        """
        total_requests = 10
        context_ids = self.get_context_ids()
        if len(context_ids) < total_requests:
            raise ValueError(
                f"Need at least {total_requests} contexts for increasing_length mode; "
                f"found {len(context_ids)}."
            )

        short_questions = self.rng.sample(QUESTION_TEMPLATES, 6)
        medium_questions = self.rng.sample(QUESTION_TEMPLATES_MEDIUM, 3)
        large_questions = self.rng.sample(QUESTION_TEMPLATES_LARGE, 1)
        questions = short_questions + medium_questions + large_questions

        selected_contexts = self.rng.sample(context_ids, total_requests)
        requests: List[Request] = []
        for i, (ctx_id, question) in enumerate(zip(selected_contexts, questions)):
            context_text = self.contexts[ctx_id]
            context_tokens = self._count_tokens(context_text)
            question_tokens = self._count_tokens(question)
            requests.append(
                Request(
                    request_id=i,
                    context_id=ctx_id,
                    context_text=context_text,
                    question=question,
                    context_tokens=context_tokens,
                    question_tokens=question_tokens,
                    token_length=context_tokens + question_tokens,
                )
            )

        return requests

    def generate_same_context_same_question(
        self, context_id: str, question: str | None = None, num_requests: int = 5
    ) -> List[Request]:
        """
        Base function: Generates N requests for the exact same context/question pair.
        """
        if context_id not in self.contexts:
            raise ValueError(f"Unknown context_id '{context_id}'")

        selected_question = question or self.rng.choice(self.questions)
        ctx_text = self.contexts[context_id]
        
        context_tokens = self._count_tokens(ctx_text)
        question_tokens = self._count_tokens(selected_question)
        
        requests = []
        for i in range(num_requests):
            requests.append(
                Request(
                    request_id=i, # Note: IDs are local to this function's scope
                    context_id=context_id,
                    context_text=ctx_text,
                    question=selected_question,
                    context_tokens=context_tokens,
                    question_tokens=question_tokens,
                    token_length=context_tokens + question_tokens,
                )
            )
        return requests

    def generate_same_context_multiple_questions(
        self, 
        context_id: str, 
        num_questions: int = 3,
        questions_list: List[str] | None = None
    ) -> List[Request]:
        """
        Calls generate_same_context_same_question for multiple different questions 
        within one context.
        It is rewritting the id counter in case the "generate_same_context_same_question
        is being called multiple times (as requests will have the same request_id)
        """
        # Default to sampling from global templates if no list provided
        if questions_list is None:
            questions_list = self.rng.sample(self.questions, min(num_questions, len(self.questions)))
        
        all_requests = []
        global_id_counter = 0
        
        for q in questions_list:
            # We call the 'same_question' function with num_requests=1
            req_list = self.generate_same_context_same_question(context_id, question=q, num_requests=5)
            # Re-assign ID to maintain sequence in this specific function's return list
            for r in req_list:
                r.request_id = global_id_counter
                all_requests.append(r)
                global_id_counter += 1
                
        return all_requests

    def generate_multiple_contexts_multiple_questions(
        self, 
        num_contexts: int = 3, 
        context_ids: List[str] | None = None
    ) -> List[Request]:
        """
        Highest level: Calls generate_same_context_multiple_questions for 
        different contexts.
        It is rewritting the id counter in case the "generate_same_context_same_question
        is being called multiple times (as requests will have the same request_id)
        """
        # Default to sampling from available context files if no list provided
        if context_ids is None:
            available = self.get_context_ids()
            context_ids = self.rng.sample(available, min(num_contexts, len(available)))

        all_requests = []
        global_id_counter = 0

        for ctx_id in context_ids:
            # Calls the multiple-question function for this context
            req_list = self.generate_same_context_multiple_questions(ctx_id)
            # Re-assign ID to maintain sequence across all contexts
            for r in req_list:
                r.request_id = global_id_counter
                all_requests.append(r)
                global_id_counter += 1

        return all_requests


# -----------------------------------------------------------------------
# Standalone usage — quick sanity check
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Request Generator — preview")
    parser.add_argument("--data-dir", default="data/", help="Context folder")
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["random", "sequential", "repeated"],
        default="random",
        help="Generation mode",
    )
    parser.add_argument(
        "--context-id",
        default=None,
        help="Context ID for 'repeated' mode",
    )
    args = parser.parse_args()

    gen = RequestGenerator(data_dir=args.data_dir, seed=args.seed)

    print(f"Loaded {len(gen.contexts)} contexts: {gen.get_context_ids()}\n")

    if args.mode == "random":
        requests = gen.generate(args.num_requests)
    elif args.mode == "sequential":
        requests = gen.generate_sequential(num_per_context=3)
    elif args.mode == "repeated":
        ctx = args.context_id or gen.get_context_ids()[0]
        requests = gen.generate_repeated(ctx, args.num_requests)

    print(f"Generated {len(requests)} requests:\n")
    print(f"{'ID':>4}  {'Context':<35}  {'Tokens':>7}  Question")
    print("-" * 100)
    for req in requests:
        print(
            f"{req.request_id:>4}  {req.context_id:<35}  "
            f"{req.token_length:>7}  {req.question[:50]}"
        )
