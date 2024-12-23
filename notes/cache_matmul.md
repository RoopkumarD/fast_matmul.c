# Cache Optimised Naive Matrix Multiplication

Minimize cache misses by exploiting spatial locality. When accessing matrix elements column-wise,
cache misses occur frequently, as cache lines are optimized for contiguous memory blocks. The
cache-aware algorithm reduces these misses.

## How It Works:

- Each element of the first matrix row is multiplied with the corresponding row of the second matrix.
- The results are accumulated into the appropriate position in the output matrix.

## Example (2x2 Matrices):

```
Input: Matrix A Matrix B [[a, b], [[e, f], [c, d]] [g, h]]

Step-by-Step:

Take a (row 1, col 1 of A) and multiply it with row 1 of B: [[ae, af], [0, 0]]

Take b (row 1, col 2 of A) and multiply it with row 2 of B, accumulating: [[ae + bg, af + bh], [0, 0]]

Repeat for row 2 of A: [[ae + bg, af + bh], [ce + dg, cf + dh]]
```

## Benchmark Comparison

Below is a benchmark comparing the naive and cache-optimized implementations. The results show that
for small matrices, both methods perform similarly due to minimal cache misses. As matrix sizes
grow, the cache-optimized implementation significantly outperforms the naive one.

![Benchmark comparison between naive and cache implementation](../images/benchmark.png)
