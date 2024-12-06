import matplotlib.pyplot as plt
import pandas as pd

MIN_SIZE = 100
MAX_SIZE = 500
NUM_PTS = 10

if __name__ == "__main__":
    naive_file = pd.read_csv(
        "naive_benchmark_results.txt", delim_whitespace=True, header=None
    )
    cache_file = pd.read_csv(
        "cache_benchmark_results.txt", delim_whitespace=True, header=None
    )

    mat_sizes = naive_file[0]

    naive_min_cycles = naive_file[1]
    naive_avg_cycles = naive_file[3]

    cache_min_cycles = cache_file[1]
    cache_avg_cycles = cache_file[3]

    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot data for naive implementation
    plt.plot(mat_sizes, naive_avg_cycles, "--*", label="Naive AVERAGE")
    plt.plot(mat_sizes, naive_min_cycles, "--*", label="Naive MIN")

    # Plot data for cache-optimized implementation
    plt.plot(mat_sizes, cache_avg_cycles, "--o", label="Cache AVERAGE")
    plt.plot(mat_sizes, cache_min_cycles, "--o", label="Cache MIN")

    # Add labels, legend, and grid
    ax.set_title("Benchmark Results Comparison", fontsize=16)
    ax.set_xlabel("Matrix Size", fontsize=14)
    ax.set_ylabel("Cycles", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid()

    # Save the graph to a file
    fig.savefig("benchmark.png")
