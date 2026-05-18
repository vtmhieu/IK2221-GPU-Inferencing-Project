import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def generate_summary_plot():
    files = glob.glob('results/q2_new/q2_util_*.csv')
    if not files:
        files = glob.glob('results/q2/q2_util_*.csv')
        
    results = []

    for f in files:
        df = pd.read_csv(f)
        if 'ttft_s' not in df.columns: continue
        df = df[df["ttft_s"].notna()].reset_index(drop=True)
        if len(df) == 0: continue
        
        df["seq_pos"] = range(len(df))
        df["ctx_changed"] = df["context_id"] != df["context_id"].shift(1)
        miss_mask = df["ctx_changed"] | (df["seq_pos"] == 0)
        
        avg_miss = df[miss_mask]["ttft_s"].mean()
        avg_hit = df[~miss_mask]["ttft_s"].mean() if sum(~miss_mask) > 0 else 0
        
        util_str = os.path.basename(f).replace('q2_util_', '').replace('.csv', '')
        try:
            util = float(util_str)
            results.append({'util': util, 'avg_hit': avg_hit, 'avg_miss': avg_miss})
        except ValueError:
            pass

    if not results:
        print("No valid CSVs found.")
        return

    results.sort(key=lambda x: x['util'])
    
    utils = [r['util'] for r in results]
    avg_hits = [r['avg_hit'] * 1000 for r in results]
    avg_misses = [r['avg_miss'] * 1000 for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(utils, avg_hits, marker='o', linestyle='-', color='#27AE60', label='Avg Hit TTFT (ms)', linewidth=2)
    plt.plot(utils, avg_misses, marker='s', linestyle='--', color='#E74C3C', label='Avg Miss TTFT (ms)', linewidth=2)
    
    plt.title("Task 1 Q2: Impact of Cache Size on TTFT", fontsize=14, fontweight="bold")
    plt.xlabel("GPU Memory Utilization (Local Cache Size)", fontsize=12)
    plt.ylabel("Time to First Token (ms)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    
    output_path = "results/q2_new_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Summary plot saved to {output_path}")

if __name__ == "__main__":
    generate_summary_plot()
