// Row Major Order
//
// Same as Column Major Order but the only difference is we are
// taking one element to broadcast from A matrix than B
// Will result in same intermediate matrix as that of Column Order
// because we are matrix multiplying one vector to other thus one element
// at once looking from both direction

// Had to create buffer for C as opposed to article because my machine supports
// till SSE_4.2 thus my machine doesn't support maskload and maskstore intrinsic

#include "benchmark.h"
#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

#define DO_BENCH 1

#define MR 4
#define NR 12

void print_othermat(float *mat, int rows, int cols) {
    printf("[");
    for (int i = 0; i < rows; i++) {
        printf("%s", (i != 0) ? " [" : "[");
        for (int j = 0; j < cols; j++) {
            printf("%f", mat[i * cols + j]);
            if (j != cols - 1) {
                printf(", ");
            }
        }
        printf("%s", (i != rows - 1) ? "],\n" : "]");
    }
    printf("]\n");
    return;
}

void kernel_4x12(float *Ablock, float *Bblock, float *C, const int K) {
    __m128 a_packFloat4;
    __m128 b0_packFloat4;
    __m128 b1_packFloat4;
    __m128 b2_packFloat4;

    // loading Cbuffer matrix into registers
    __m128 C_buffer00 = _mm_loadu_ps(C);
    __m128 C_buffer01 = _mm_loadu_ps(C + 4);
    __m128 C_buffer02 = _mm_loadu_ps(C + 8);

    __m128 C_buffer10 = _mm_loadu_ps(C + NR);
    __m128 C_buffer11 = _mm_loadu_ps(C + NR + 4);
    __m128 C_buffer12 = _mm_loadu_ps(C + NR + 8);

    __m128 C_buffer20 = _mm_loadu_ps(C + 2 * NR);
    __m128 C_buffer21 = _mm_loadu_ps(C + 2 * NR + 4);
    __m128 C_buffer22 = _mm_loadu_ps(C + 2 * NR + 8);

    __m128 C_buffer30 = _mm_loadu_ps(C + 3 * NR);
    __m128 C_buffer31 = _mm_loadu_ps(C + 3 * NR + 4);
    __m128 C_buffer32 = _mm_loadu_ps(C + 3 * NR + 8);

    for (int i = 0; i < K; i++) {
        b0_packFloat4 = _mm_loadu_ps(Bblock + i * NR);
        b1_packFloat4 = _mm_loadu_ps(Bblock + i * NR + 4);
        b2_packFloat4 = _mm_loadu_ps(Bblock + i * NR + 8);

        a_packFloat4 = _mm_set1_ps(Ablock[i]);
        C_buffer00 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b0_packFloat4), C_buffer00);
        C_buffer01 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b1_packFloat4), C_buffer01);
        C_buffer02 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b2_packFloat4), C_buffer02);

        a_packFloat4 = _mm_set1_ps(Ablock[i + K]);
        C_buffer10 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b0_packFloat4), C_buffer10);
        C_buffer11 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b1_packFloat4), C_buffer11);
        C_buffer12 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b2_packFloat4), C_buffer12);

        a_packFloat4 = _mm_set1_ps(Ablock[i + K * 2]);
        C_buffer20 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b0_packFloat4), C_buffer20);
        C_buffer21 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b1_packFloat4), C_buffer21);
        C_buffer22 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b2_packFloat4), C_buffer22);

        a_packFloat4 = _mm_set1_ps(Ablock[i + K * 3]);
        C_buffer30 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b0_packFloat4), C_buffer30);
        C_buffer31 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b1_packFloat4), C_buffer31);
        C_buffer32 =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b2_packFloat4), C_buffer32);
    }

    _mm_storeu_ps(C, C_buffer00);
    _mm_storeu_ps(C + 4, C_buffer01);
    _mm_storeu_ps(C + 8, C_buffer02);

    _mm_storeu_ps(C + NR, C_buffer10);
    _mm_storeu_ps(C + NR + 4, C_buffer11);
    _mm_storeu_ps(C + NR + 8, C_buffer12);

    _mm_storeu_ps(C + 2 * NR, C_buffer20);
    _mm_storeu_ps(C + 2 * NR + 4, C_buffer21);
    _mm_storeu_ps(C + 2 * NR + 8, C_buffer22);

    _mm_storeu_ps(C + 3 * NR, C_buffer30);
    _mm_storeu_ps(C + 3 * NR + 4, C_buffer31);
    _mm_storeu_ps(C + 3 * NR + 8, C_buffer32);

    return;
}

void kernel_matmul(matrix *A, matrix *B, matrix *out) {
    const int M = A->rows;
    const int N = B->cols;
    const int K = B->rows;

    float Abuffer[MR * K];
    float Bbuffer[K * NR];
    float Cbuffer[MR * NR];

    /*
     * Reason for making J loop outer because of the filling of Bbuffer
     * Here, while filling Bbuffer we are having approximately K cache misses
     * Since outer loop filling will result in K * (N / NR) cache misses totally
     *
     * Otherwise if this was inside then cache misses would have been
     * K * (N / NR) * (M / MR)
     *
     * Here, having J loop inside will lead to having alot of cache references
     * -> use perf to check
     *  perf -e cache-references,cache-misses exec.out
     */

    for (int j = 0; j < N; j += NR) {
        int n = ((N - j) < NR) ? (N - j) : NR;

        // not conditionally filling zero for last part
        // as this is fast compared to memset(..,0,(n < NR) * ...);
        memset(Bbuffer, 0, sizeof(float) * K * NR);
        for (int nk = 0; nk < K; nk++) {
            memcpy(Bbuffer + nk * NR, B->data + nk * N + j, sizeof(float) * n);
        }

        for (int i = 0; i < M; i += MR) {
            int m = ((M - i) < MR) ? (M - i) : MR;

            memcpy(Abuffer, A->data + i * K, sizeof(float) * K * m);
            memset(Abuffer + m * K, 0, sizeof(float) * K * (MR - m));

            memset(Cbuffer, 0, sizeof(float) * MR * NR);
            for (int cm = 0; cm < m; cm++) {
                memcpy(Cbuffer + cm * NR, out->data + i * N + j + cm * N,
                       sizeof(float) * n);
            }

            kernel_4x12(Abuffer, Bbuffer, Cbuffer, K);

            // storing result in out
            for (int cm = 0; cm < m; cm++) {
                memcpy(out->data + i * N + j + cm * N, Cbuffer + cm * NR,
                       sizeof(float) * n);
            }
        }
    }

    return;
}

void compare_matrix(matrix *a, matrix *b) {
    int total_data = a->total_data;
    for (int i = 0; i < total_data; i++) {
        if (a->data[i] != b->data[i]) {
            fprintf(stderr, "Didn't Match at [%d][%d]\n", i / a->cols,
                    i % a->cols);
            return;
        }
    }
    printf("Match!!\n");
    return;
}

int main(void) {
    srand(time(NULL));

#ifdef DO_BENCH
    benchmark(kernel_matmul, "benchmarks/4x12_row_optimise.txt");
#else
    const int M = rand() % 2000;
    const int N = rand() % 2000;
    const int K = rand() % 2000;

    if (M == 0 || N == 0 || K == 0) {
        printf("Got zero as dimension\n");
        return 1;
    }
    printf("M = %d, N = %d, K = %d\n", M, N, K);

    matrix *a = mat_create(M, K);
    if (a == NULL) {
        perror("Allocation Failed for a");
        goto cleanup;
    }
    matrix *b = mat_create(K, N);
    if (b == NULL) {
        perror("Allocation Failed for b");
        goto cleanup;
    }
    matrix *c = mat_create(M, N);
    if (c == NULL) {
        perror("Allocation Failed for c");
        goto cleanup;
    }
    matrix *d = mat_create(M, N);
    if (d == NULL) {
        perror("Allocation Failed for d");
        goto cleanup;
    }

    fill_random(a);
    fill_random(b);
    memset(c->data, 0, sizeof(float) * c->total_data);
    memset(d->data, 0, sizeof(float) * d->total_data);

    kernel_matmul(a, b, c);
    printf("Done with kernel matmul\n");
    naive_matmul(a, b, d);

    compare_matrix(c, d);

cleanup:
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(d);
#endif
    return 0;
}
