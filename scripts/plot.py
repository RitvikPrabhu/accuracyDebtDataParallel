#!/usr/bin/env python
"""
Manually plot Log SAD Score and Time vs GPU World Size.
"""
import matplotlib.pyplot as plt

def get_data():
    """
    MANUAL DATA ENTRY SECTION
    Format:  gpu_count: (time_in_seconds, log_sad_score)
    Use None if you don't have a value (e.g., (2100, None))
    """
    data = {
        "arar": {
            1: (2146.32, 5.267858159578791),
            2: (2111.89, 5.267858159578791),
            3: (2122.30, 5.267858159578791),
            4: (2152.03, 5.267858159578791),
        },
        "ddp": {
            1: (2043.23, 21.15306854248047),
            2: (2053.99, 20.81602668762207),
            3: (2052.64, 21.15306854248047),
            4: (2050.32, 20.81602668762207),
            8: (2049.48, 21.07143211364746),
        },
        "hvd": {
            1: (2134.27, 21.02998161315918),
            2: (2267.65, 21.07143211364746),
            3: (2298.35, 21.02998161315918),
            4: (2576.60, 21.02998161315918),
            8: (2407.85, 20.859743118286133),
        }
    }
    return data

def plot(data, out_path="manual_performance_metrics.png"):
    methods = ["arar", "ddp", "hvd"]
    colors = {"arar": "C0", "ddp": "C1", "hvd": "C2"}

    # Create 2 Subplots
    fig, (ax_time, ax_sad) = plt.subplots(1, 2, figsize=(14, 6))

    for method in methods:
        if method not in data:
            continue
            
        # Extract and sort points by GPU count (x-axis)
        points = sorted(data[method].items()) # [(gpu, (time, sad)), ...]
        
        if not points:
            continue

        xs = [p[0] for p in points]
        ys_time = [p[1][0] for p in points]
        ys_sad = [p[1][1] for p in points]

        # Plot Time (Speed) - Dashed Line
        # Filter out None values
        valid_time = [(x, y) for x, y in zip(xs, ys_time) if y is not None]
        if valid_time:
            vt_x, vt_y = zip(*valid_time)
            ax_time.plot(vt_x, vt_y, marker="s", linestyle="--", color=colors[method], label=method.upper())

        # Plot Accuracy (Log SAD) - Solid Line
        valid_sad = [(x, y) for x, y in zip(xs, ys_sad) if y is not None]
        if valid_sad:
            vs_x, vs_y = zip(*valid_sad)
            ax_sad.plot(vs_x, vs_y, marker="o", linestyle="-", color=colors[method], label=method.upper())

    ax_time.set_title("Training Speed vs GPU Count")
    ax_time.set_xlabel("World Size (GPUs)")
    ax_time.set_ylabel("Time to Solution (seconds)")
    ax_time.grid(True, linestyle="--", alpha=0.4)
    ax_time.legend()

    ax_sad.set_title("Model Convergence vs GPU Count")
    ax_sad.set_xlabel("World Size (GPUs)")
    ax_sad.set_ylabel("Log SAD Score")
    ax_sad.grid(True, linestyle="--", alpha=0.4)
    ax_sad.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plots to {out_path}")

def main():
    data = get_data()
    plot(data)

if __name__ == "__main__":
    main()