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
from dataclasses import dataclass, field
from typing import List, Iterator


# ---------------------------------------------------------------------------
# Pre-defined question templates — each is generic enough to apply to any
# research-paper context.  Extend this list to increase diversity.
# ---------------------------------------------------------------------------
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
]


@dataclass
class Request:
    """A single benchmark request."""
    request_id: int
    context_id: str          # filename stem, e.g. "vllm"
    context_text: str        # full document content
    question: str            # question to ask about the context
    seq_length: int = 0      # context + question character count (set after init)

    def __post_init__(self):
        self.seq_length = len(self.context_text) + len(self.question)


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
        questions: List[str] | None = None,
        seed: int | None = 42,
    ):
        self.data_dir = data_dir
        self.questions = questions or QUESTION_TEMPLATES
        self.rng = random.Random(seed)
        self.contexts = self._load_contexts()

        if not self.contexts:
            raise FileNotFoundError(
                f"No .txt files found in '{data_dir}'. "
                "Make sure context files are present."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_context_ids(self) -> List[str]:
        """Return the list of all loaded context IDs."""
        return list(self.contexts.keys())

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
            requests.append(
                Request(
                    request_id=i,
                    context_id=ctx_id,
                    context_text=self.contexts[ctx_id],
                    question=question,
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
                requests.append(
                    Request(
                        request_id=req_id,
                        context_id=ctx_id,
                        context_text=ctx_text,
                        question=question,
                    )
                )
                req_id += 1
        return requests

    def generate_repeated(
        self, context_id: str, num_requests: int = 10
    ) -> List[Request]:
        """
        Generate multiple requests for the *same* context.

        Useful for Task 1 Q2: measuring latency improvement when a
        previously-seen context is fed again (KV cache reuse).
        """
        if context_id not in self.contexts:
            raise ValueError(
                f"Unknown context_id '{context_id}'. "
                f"Available: {self.get_context_ids()}"
            )
        ctx_text = self.contexts[context_id]
        requests = []
        for i in range(num_requests):
            requests.append(
                Request(
                    request_id=i,
                    context_id=context_id,
                    context_text=ctx_text,
                    question=self.rng.choice(self.questions),
                )
            )
        return requests


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
    print(f"{'ID':>4}  {'Context':<35}  {'SeqLen':>7}  Question")
    print("-" * 100)
    for req in requests:
        print(
            f"{req.request_id:>4}  {req.context_id:<35}  "
            f"{req.seq_length:>7}  {req.question[:50]}"
        )
