"""
plot_q2.py — Task 1 Q2: KV Cache Reuse Visualisation

Reads a CSV produced by:
    python benchmark.py --mode repeated --num-contexts N -o results/q2.csv

and generates a clean plot that makes the cache-miss / cache-hit pattern
immediately obvious.

Usage:
    python plot_q2.py                                        # default paths
    python plot_q2.py --input results/q2.csv --output results/q2.png
"""

import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# ── colour palette for context bands ──────────────────────────────────────────
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

MISS_COLOR = "#E74C3C"   # red  — cache miss (context change)
HIT_COLOR  = "#27AE60"   # green — question change within same context
LINE_COLOR = "#2C3E50"   # dark blue-grey — TTFT line


# ── data loading ──────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop rows where TTFT measurement failed
    df = df[df["ttft_s"].notna()].reset_index(drop=True)

    # Use row position as x-axis — more reliable than stored request_id
    # (avoids issues if IDs were reset or requests were skipped)
    df["seq_pos"] = range(len(df))

    # Detect group boundaries
    df["ctx_changed"] = df["context_id"] != df["context_id"].shift(1)
    df["q_changed"]   = df["question"]   != df["question"].shift(1)

    return df


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_cache_reuse(csv_path: str, output_path: str) -> None:
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = load_and_prepare(csv_path)

    unique_contexts = df["context_id"].unique()
    ctx_color_map = {
        ctx: PALETTE[i % len(PALETTE)]
        for i, ctx in enumerate(unique_contexts)
    }

    # ── figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    # ── shaded bands — one colour per context ─────────────────────────────────
    ctx_boundaries = [0] + df.index[df["ctx_changed"]].tolist() + [len(df)]
    for i in range(len(ctx_boundaries) - 1):
        start = ctx_boundaries[i]
        end   = ctx_boundaries[i + 1] - 1
        ctx   = df.loc[start, "context_id"]
        ax.axvspan(start - 0.5, end + 0.5,
                   color=ctx_color_map[ctx], alpha=0.10, zorder=0)

    # ── dotted vertical lines at context boundaries ───────────────────────────
    for idx in df.index[df["ctx_changed"] & (df["seq_pos"] > 0)]:
        ax.axvline(idx - 0.5, color="#95A5A6", linewidth=0.9,
                   linestyle=":", alpha=0.8, zorder=1)

    # ── TTFT line ─────────────────────────────────────────────────────────────
    ax.plot(df["seq_pos"], df["ttft_s"],
            color=LINE_COLOR, linewidth=1.3, alpha=0.75, zorder=2, label="TTFT")

    # ── cache-miss markers — first request of each new context ────────────────
    miss_mask = df["ctx_changed"] | (df["seq_pos"] == 0)
    miss_rows = df[miss_mask]
    ax.scatter(miss_rows["seq_pos"], miss_rows["ttft_s"],
               color=MISS_COLOR, s=90, zorder=5,
               edgecolors="white", linewidth=1.3,
               label="Cache miss (context change)")

    # ── question-change markers — within the same context ─────────────────────
    q_only_mask = df["q_changed"] & ~df["ctx_changed"] & (df["seq_pos"] > 0)
    q_only_rows = df[q_only_mask]
    ax.scatter(q_only_rows["seq_pos"], q_only_rows["ttft_s"],
               color=HIT_COLOR, s=70, marker="^", zorder=4,
               edgecolors="white", linewidth=1.0,
               label="Question change (same context)")

    # ── average TTFT reference lines ──────────────────────────────────────────
    hit_rows = df[~miss_mask]
    if len(hit_rows):
        avg_hit = hit_rows["ttft_s"].mean()
        ax.axhline(avg_hit, color=HIT_COLOR, linewidth=1.3, linestyle="--",
                   alpha=0.8,
                   label=f"Avg cache-hit TTFT  ({avg_hit * 1000:.1f} ms)")

    if len(miss_rows):
        avg_miss = miss_rows["ttft_s"].mean()
        ax.axhline(avg_miss, color=MISS_COLOR, linewidth=1.3, linestyle="--",
                   alpha=0.8,
                   label=f"Avg cache-miss TTFT ({avg_miss * 1000:.1f} ms)")

    # ── legend: behaviour markers ─────────────────────────────────────────────
    legend1 = ax.legend(loc="upper right", frameon=True, fontsize=9,
                        framealpha=0.92, edgecolor="#DEE2E6")

    # ── legend: context colour patches ────────────────────────────────────────
    ctx_patches = [
        mpatches.Patch(color=ctx_color_map[ctx], alpha=0.6, label=ctx)
        for ctx in unique_contexts
    ]
    legend2 = ax.legend(handles=ctx_patches, loc="upper left",
                        title="Context", fontsize=8, frameon=True,
                        framealpha=0.92, edgecolor="#DEE2E6", title_fontsize=9)
    ax.add_artist(legend1)  # re-add first legend (add_artist prevents override)

    # ── axes styling ──────────────────────────────────────────────────────────
    ax.set_title("Task 1 Q2: TTFT and KV Cache Reuse Analysis — KV cache reuse analysis - Max local cache size = 0.8",
                 fontsize=14, fontweight="bold", pad=14, color="#2C3E50")
    ax.set_xlabel("Request Sequence Position", fontsize=12, color="#2C3E50")
    ax.set_ylabel("Time to First Token (s)",   fontsize=12, color="#2C3E50")
    ax.set_xlim(-1, len(df))
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.5, color="#BDC3C7")
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#DEE2E6")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Task 1 Q2"
    )
    parser.add_argument("--input",  default="results/q2_util_0.8.csv",
                        help="Input CSV from benchmark.py --mode repeated")
    parser.add_argument("--output", default="results/q2_util_0.8.png",
                        help="Output PNG path")
    args = parser.parse_args()
    plot_cache_reuse(args.input, args.output)


if __name__ == "__main__":
    main()