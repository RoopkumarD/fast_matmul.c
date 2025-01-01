#include "matrix.h"

#ifndef BENCHMARK_H
#define BENCHMARK_H 1

void single_bench(void matmul(matrix *, matrix *, matrix *));
void benchmark(void matmul(matrix *a, matrix *b, matrix *out), char *filename);

#endif // BENCHMARK_H
