// Since i have SSE instruction set -> thus 16 XMM registers and no FMA
// instructions So i will do the both FMA in two instruction
//
// Column Major Order

#include "cmk.h"
#include "matrix.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

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

void kernel_12x4(float *Ablock, float *Bblock, float *C, const int M,
                 const int N, const int K) {
    __m128 C_buffer[4][3]; // taking nR = 4 and mR = 12
    __m128 b_packFloat4;
    __m128 a0_packFloat4;
    __m128 a1_packFloat4;
    __m128 a2_packFloat4;

    for (int i = 0; i < 4; i++) {
        C_buffer[i][0] = _mm_loadu_ps(C + i * M);
        C_buffer[i][1] = _mm_loadu_ps(C + i * M + 4);
        C_buffer[i][2] = _mm_loadu_ps(C + i * M + 8);
    }

    for (int p = 0; p < K; p++) {
        a0_packFloat4 = _mm_loadu_ps(Ablock + p * M);
        a1_packFloat4 = _mm_loadu_ps(Ablock + p * M + 4);
        a2_packFloat4 = _mm_loadu_ps(Ablock + p * M + 8);

        b_packFloat4 = _mm_set1_ps(Bblock[p]);
        C_buffer[0][0] =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer[0][0]);
        C_buffer[0][1] =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer[0][1]);
        C_buffer[0][2] =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer[0][2]);

        b_packFloat4 = _mm_set1_ps(Bblock[p + K * 1]);
        C_buffer[1][0] =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer[1][0]);
        C_buffer[1][1] =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer[1][1]);
        C_buffer[1][2] =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer[1][2]);

        b_packFloat4 = _mm_set1_ps(Bblock[p + K * 2]);
        C_buffer[2][0] =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer[2][0]);
        C_buffer[2][1] =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer[2][1]);
        C_buffer[2][2] =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer[2][2]);

        b_packFloat4 = _mm_set1_ps(Bblock[p + K * 3]);
        C_buffer[3][0] =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer[3][0]);
        C_buffer[3][1] =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer[3][1]);
        C_buffer[3][2] =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer[3][2]);
    }

    for (int j = 0; j < 4; j++) {
        _mm_storeu_ps(C + j * M, C_buffer[j][0]);
        _mm_storeu_ps(C + j * M + 4, C_buffer[j][1]);
        _mm_storeu_ps(C + j * M + 8, C_buffer[j][2]);
    }

    return;
}

void kernel_matmul(matrix *A, matrix *B, matrix *out) {
    int M = A->rows;
    int N = B->cols;
    int K = B->rows;

    float Abuffer[MR * K];
    float Bbuffer[K * NR];
    float Cbuffer[MR * NR];

    for (int i = 0; i < M; i += 12) {
        int m = ((M - i) < MR) ? (M - i) : MR;

        for (int mk = 0; mk < K; mk++) {
            memcpy(Abuffer + mk * MR, A->data + mk * M + i, sizeof(float) * m);
            memset(Abuffer + mk * MR + m, 0, sizeof(float) * (MR - m));
        }

        for (int j = 0; j < N; j += 4) {
            int n = ((N - j) < NR) ? (N - j) : NR;

            memcpy(Bbuffer, B->data + j * K, sizeof(float) * K * n);
            memset(Bbuffer + n * K, 0, sizeof(float) * K * (NR - n));

            int cn = 0;
            for (; cn < n; cn++) {
                memcpy(Cbuffer + cn * MR, out->data + j * M + i + cn * M,
                       sizeof(float) * m);
                memset(Cbuffer + cn * MR + m, 0, sizeof(float) * (MR - m));
            }
            memset(Cbuffer + cn * MR, 0, sizeof(float) * MR * (NR - cn));

            kernel_12x4(Abuffer, Bbuffer, Cbuffer, MR, NR, K);

            // storing result in out
            for (int cn = 0; cn < n; cn++) {
                memcpy(out->data + j * M + i + cn * M, Cbuffer + cn * MR,
                       sizeof(float) * m);
            }
        }
    }
    return;
}

int main2(void) {
    srand(time(NULL));

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
    return 0;
}
