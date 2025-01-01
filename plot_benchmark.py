import matplotlib.pyplot as plt
import pandas as pd
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception("Usage: python3 plot.py data1.txt ...")

    plt.rc("font", size=10)
    fig, ax = plt.subplots(figsize=(8, 6))

    for plot in sys.argv[1:]:
        label = plot.split("/")[-1].split(".")[0]
        benchmark_result = pd.read_csv(
            plot, delim_whitespace=True, header=None
        )
        mat_sizes = benchmark_result[0]
        benchmark_avg_flops = benchmark_result[3]
        plt.plot(mat_sizes, benchmark_avg_flops, f"--*", label=f"{label}")

    # Add labels, legend, and grid
    ax.set_title("Benchmark Results Comparison", fontsize=10)
    ax.set_xlabel("Matrix Size", fontsize=8)
    ax.set_ylabel("FLOPS", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid()

    # Save the graph to a file
    fig.savefig("benchmarks/benchmark.png")
