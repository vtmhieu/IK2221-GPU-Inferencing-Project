import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_cache_reuse(csv_path, output_path):
    # load the data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # get the transition points to plot the graph nicelly
    df['context_changed'] = df['context_id'] != df['context_id'].shift(1)
    df['question_changed'] = df['question'] != df['question'].shift(1)

    # exclude the first modification (from nothing to first context and question)
    context_changes = df[df['context_changed'] & (df.index > 0)]
    question_changes = df[df['question_changed'] & (df.index > 0)]

    # create the plot
    plt.figure(figsize=(14, 7))

    # plot the TTFT line
    plt.plot(df['request_id'], df['ttft_s'], color='#2c3e50', linewidth=1.5, 
             alpha=0.6, label='TTFT (seconds)', zorder=1)

    # Plot Question Changes (Green)
    plt.scatter(question_changes['request_id'], question_changes['ttft_s'], 
                color='green', label='Question Change', s=60, 
                edgecolors='white', linewidth=1, zorder=2)

    # Plot Context Changes (red)
    # Larger size so it's visible if both change simultaneously
    plt.scatter(context_changes['request_id'], context_changes['ttft_s'], 
                color='red', label='Context Change', s=100, 
                edgecolors='white', linewidth=1.5, zorder=3)

    # formatting
    plt.title('Task 1 Q2: TTFT and KV Cache Reuse Analysis', fontsize=14, pad=15)
    plt.xlabel('Request Sequence ID', fontsize=12)
    plt.ylabel('Time to First Token (s)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(frameon=True, shadow=True)

    plt.ylim(bottom=0)

    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved plot to: {output_path}")

if __name__ == "__main__":
    INPUT_CSV = "results/q2_cache_reuse.csv" 
    OUTPUT_PNG = "results/q2_cache_reuse.png"
    
    plot_cache_reuse(INPUT_CSV, OUTPUT_PNG)