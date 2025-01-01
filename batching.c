#include "benchmark.h"
#include "matrix.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

#define DO_BENCH_SINGLE 1

#define MR 12
#define NR 3

void compare_mats(float *mat1, float *mat2, const int M, const int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabsf(mat1[j * M + i] - mat2[j * M + i]) > 1e-3) {
                printf("MISMATCH! Element[%d][%d] %f != %f\n", i, j,
                       mat1[j * M + i], mat2[j * M + i]);
                return;
            }
        }
    }
    printf("MATCH!\n");
    return;
}

void matrix_printc(matrix *mat) {
    int cols = mat->cols;
    int rows = mat->rows;
    printf("[");
    for (int i = 0; i < rows; i++) {
        printf("%s", (i != 0) ? " [" : "[");
        for (int j = 0; j < cols; j++) {
            printf("%f", mat->data[j * rows + i]);
            if (j != cols - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i != rows - 1) {
            printf(",\n");
        }
    }
    printf("]\n");
    return;
}

void matmul_naivec(matrix *a, matrix *b, matrix *c) {
    int M = a->rows;
    int N = b->cols;
    int K = b->rows;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int p = 0; p < K; p++) {
                c->data[j * M + i] += a->data[p * M + i] * b->data[j * K + p];
            }
        }
    }
}

void kernel_12x4(float *Ablock, float *Bblock, float *C, const int K) {
    __m128 b_packFloat4;
    __m128 a0_packFloat4;
    __m128 a1_packFloat4;
    __m128 a2_packFloat4;

    // loading Cbuffer matrix into registers
    __m128 C_buffer00 = _mm_loadu_ps(C);
    __m128 C_buffer01 = _mm_loadu_ps(C + 4);
    __m128 C_buffer02 = _mm_loadu_ps(C + 8);

    __m128 C_buffer10 = _mm_loadu_ps(C + MR);
    __m128 C_buffer11 = _mm_loadu_ps(C + MR + 4);
    __m128 C_buffer12 = _mm_loadu_ps(C + MR + 8);

    __m128 C_buffer20 = _mm_loadu_ps(C + 2 * MR);
    __m128 C_buffer21 = _mm_loadu_ps(C + 2 * MR + 4);
    __m128 C_buffer22 = _mm_loadu_ps(C + 2 * MR + 8);

    for (int p = 0; p < K; p++) {
        a0_packFloat4 = _mm_loadu_ps(Ablock + p * MR);
        a1_packFloat4 = _mm_loadu_ps(Ablock + p * MR + 4);
        a2_packFloat4 = _mm_loadu_ps(Ablock + p * MR + 8);

        b_packFloat4 = _mm_set1_ps(Bblock[p]);
        C_buffer00 =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer00);
        C_buffer01 =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer01);
        C_buffer02 =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer02);

        b_packFloat4 = _mm_set1_ps(Bblock[p + K * 1]);
        C_buffer10 =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer10);
        C_buffer11 =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer11);
        C_buffer12 =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer12);

        b_packFloat4 = _mm_set1_ps(Bblock[p + K * 2]);
        C_buffer20 =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer20);
        C_buffer21 =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer21);
        C_buffer22 =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer22);
    }

    _mm_storeu_ps(C, C_buffer00);
    _mm_storeu_ps(C + 4, C_buffer01);
    _mm_storeu_ps(C + 8, C_buffer02);

    _mm_storeu_ps(C + MR, C_buffer10);
    _mm_storeu_ps(C + MR + 4, C_buffer11);
    _mm_storeu_ps(C + MR + 8, C_buffer12);

    _mm_storeu_ps(C + 2 * MR, C_buffer20);
    _mm_storeu_ps(C + 2 * MR + 4, C_buffer21);
    _mm_storeu_ps(C + 2 * MR + 8, C_buffer22);

    return;
}

void kernel_matmul(matrix *A, matrix *B, matrix *out) {
    int M = A->rows;
    int N = B->cols;
    int K = B->rows;

    float Abuffer[MR * K];
    float Bbuffer[K * NR];
    float Cbuffer[MR * NR];

    for (int i = 0; i < M; i += MR) {
        int m = ((M - i) < MR) ? (M - i) : MR;

        // rather than memset(Ab, 0, (m < MR) * ...); this
        // as this is always fast
        memset(Abuffer, 0, sizeof(float) * MR * K);
        for (int mk = 0; mk < K; mk++) {
            memcpy(Abuffer + mk * MR, A->data + mk * M + i, sizeof(float) * m);
        }

#pragma omp parallel for
        for (int j = 0; j < N; j += NR) {
            int n = ((N - j) < NR) ? (N - j) : NR;

            memcpy(Bbuffer, B->data + j * K, sizeof(float) * K * n);
            memset(Bbuffer + n * K, 0, sizeof(float) * K * (NR - n));

            // no conditional mask as above explaned for Abuffer
            memset(Cbuffer, 0, sizeof(float) * MR * NR);
            for (int cn = 0; cn < n; cn++) {
                memcpy(Cbuffer + cn * MR, out->data + j * M + i + cn * M,
                       sizeof(float) * m);
            }

            kernel_12x4(Abuffer, Bbuffer, Cbuffer, K);

            // storing result in out
            for (int cn = 0; cn < n; cn++) {
                memcpy(out->data + j * M + i + cn * M, Cbuffer + cn * MR,
                       sizeof(float) * m);
            }
        }
    }
    return;
}

int main(void) {
    srand(time(NULL));

#ifdef DO_BENCH
    char filename[100];
    memset(filename, 0, 100 * sizeof(char));
    sprintf(filename, "benchmarks/%dx%d_column.txt", MR, NR);
    benchmark(kernel_matmul, filename);
#elif DO_BENCH_SINGLE
    single_bench(kernel_matmul);
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
    matrix *b = mat_create(K, N);
    matrix *c = mat_create(M, N);
    matrix *d = mat_create(M, N);

    fill_random(a);
    fill_random(b);
    memset(c->data, 0, c->total_data * sizeof(float));
    memset(d->data, 0, d->total_data * sizeof(float));

    kernel_matmul(a, b, c);
    printf("Done with kernel matmul\n");
    matmul_naivec(a, b, d);

    compare_mats(c->data, d->data, M, N);

    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(d);
#endif // DO_BENCH

    return 0;
}
