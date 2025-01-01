// Since i have SSE instruction set -> thus 16 XMM registers and no FMA
// instructions So i will do the both FMA in two instruction
//
// Column Major Order

#include "benchmark.h"
#include "matrix.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

// #define DO_BENCH_SINGLE 1
#define DO_BENCH 1

#define MR 12
#define NR 3

#define MC 24  // MR * 2
#define NC 288 // NR * 96
#define KC 2000

static float Abuffer[MC * KC];
static float Bbuffer[KC * NC];
static float Cbuffer[MC * NC];

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

void matrix_printc(float *mat, int rows, int cols) {
    printf("[");
    for (int i = 0; i < rows; i++) {
        printf("%s", (i != 0) ? " [" : "[");
        for (int j = 0; j < cols; j++) {
            printf("%f", mat[j * rows + i]);
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

void kernel_12x3(float *Ablock, float *Bblock, float *C) {
    __m128 b_packFloat4;
    __m128 a0_packFloat4;
    __m128 a1_packFloat4;
    __m128 a2_packFloat4;

    // loading Cbuffer matrix into registers
    __m128 C_buffer00 = _mm_loadu_ps(C);
    __m128 C_buffer01 = _mm_loadu_ps(C + 4);
    __m128 C_buffer02 = _mm_loadu_ps(C + 8);

    __m128 C_buffer10 = _mm_loadu_ps(C + MC);
    __m128 C_buffer11 = _mm_loadu_ps(C + MC + 4);
    __m128 C_buffer12 = _mm_loadu_ps(C + MC + 8);

    __m128 C_buffer20 = _mm_loadu_ps(C + 2 * MC);
    __m128 C_buffer21 = _mm_loadu_ps(C + 2 * MC + 4);
    __m128 C_buffer22 = _mm_loadu_ps(C + 2 * MC + 8);

    for (int p = 0; p < KC; p++) {
        a0_packFloat4 = _mm_loadu_ps(Ablock + p * MC);
        a1_packFloat4 = _mm_loadu_ps(Ablock + p * MC + 4);
        a2_packFloat4 = _mm_loadu_ps(Ablock + p * MC + 8);

        b_packFloat4 = _mm_set1_ps(Bblock[p]);
        C_buffer00 =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer00);
        C_buffer01 =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer01);
        C_buffer02 =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer02);

        b_packFloat4 = _mm_set1_ps(Bblock[p + KC * 1]);
        C_buffer10 =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer10);
        C_buffer11 =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer11);
        C_buffer12 =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer12);

        b_packFloat4 = _mm_set1_ps(Bblock[p + KC * 2]);
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

    _mm_storeu_ps(C + MC, C_buffer10);
    _mm_storeu_ps(C + MC + 4, C_buffer11);
    _mm_storeu_ps(C + MC + 8, C_buffer12);

    _mm_storeu_ps(C + 2 * MC, C_buffer20);
    _mm_storeu_ps(C + 2 * MC + 4, C_buffer21);
    _mm_storeu_ps(C + 2 * MC + 8, C_buffer22);

    return;
}

void kernel_matmul(matrix *A, matrix *B, matrix *out) {
    int M = A->rows;
    int N = B->cols;
    int K = B->rows;

    for (int j = 0; j < N; j += NC) {
        int nc = ((N - j) < NC) ? (N - j) : NC;

        for (int k = 0; k < K; k += KC) {
            int kc = ((K - k) < KC) ? (K - k) : KC;

            memset(Bbuffer, 0, sizeof(float) * KC * NC);
            // filling Bbuffer
#pragma omp parallel for
            for (int t = 0; t < nc; t++) {
                memcpy(Bbuffer + t * KC, B->data + t * K + k + j * K,
                       sizeof(float) * kc);
            }

            for (int i = 0; i < M; i += MC) {
                int mc = ((M - i) < MC) ? (M - i) : MC;

                // filling Abuffer
                memset(Abuffer, 0, sizeof(float) * MC * KC);
#pragma omp parallel for
                for (int t = 0; t < kc; t++) {
                    memcpy(Abuffer + t * MC, A->data + t * M + i + k * M,
                           sizeof(float) * mc);
                }

                // filling Cbuffer
                memset(Cbuffer, 0, sizeof(float) * MC * NC);
#pragma omp parallel for
                for (int t = 0; t < nc; t++) {
                    memcpy(Cbuffer + t * MC, out->data + t * M + i + j * M,
                           sizeof(float) * mc);
                }

#pragma omp parallel for collapse(2)
                for (int ir = 0; ir < MC; ir += MR) {
                    for (int jr = 0; jr < NC; jr += NR) {
                        kernel_12x3(Abuffer + ir, Bbuffer + jr * KC,
                                    Cbuffer + jr * MC + ir);
                    }
                }

                // setting C values to out
                for (int t = 0; t < nc; t++) {
                    memcpy(out->data + j * M + i + t * M, Cbuffer + t * MC,
                           sizeof(float) * mc);
                }
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
    sprintf(filename, "benchmarks/%dx%d_batched_column.txt", MR, NR);
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

    /*
    printf("\n============ A ===================\n");
    matrix_printc(a->data, a->rows, a->cols);
    printf("\n============ B ===================\n");
    matrix_printc(b->data, b->rows, b->cols);
    */

    compare_mats(c->data, d->data, M, N);

    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(d);
#endif // DO_BENCH

    return 0;
}
