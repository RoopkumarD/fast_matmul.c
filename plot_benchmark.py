import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=(8, 6))

    optimised = "benchmarks/12x4_column_optimise.txt"
    benchmark_result = pd.read_csv(
        optimised, delim_whitespace=True, header=None
    )
    mat_sizes = benchmark_result[0]
    benchmark_avg_flops = benchmark_result[3]
    benchmark_min_flops = benchmark_result[1]

    plt.plot(mat_sizes, benchmark_avg_flops, f"--*", label=f"AVERAGE 12x4 COLUMN")

    optimised = "benchmarks/4x12_row_optimise.txt"
    benchmark_result = pd.read_csv(
        optimised, delim_whitespace=True, header=None
    )
    mat_sizes = benchmark_result[0]
    benchmark_avg_flops = benchmark_result[3]
    benchmark_min_flops = benchmark_result[1]

    plt.plot(mat_sizes, benchmark_avg_flops, f"--o", label=f"AVERAGE 4x12 ROW")

    # Add labels, legend, and grid
    ax.set_title("Benchmark Results Comparison", fontsize=16)
    ax.set_xlabel("Matrix Size", fontsize=14)
    ax.set_ylabel("FLOPS", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid()

    # Save the graph to a file
    fig.savefig("benchmarks/benchmark.png")
