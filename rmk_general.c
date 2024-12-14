// Row Major Order
//
// Same as Column Major Order but the only difference is we are
// taking one element to broadcast from A matrix than B
// Will result in same intermediate matrix as that of Column Order
// because we are matrix multiplying one vector to other thus one element
// at once looking from both direction

// Had to create buffer for C as opposed to article because my machine supports
// till SSE_4.2 thus my machine doesn't support maskload and maskstore intrinsic

#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

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

void kernel_4x12(float *Ablock, float *Bblock, float *C, const int M,
                 const int N, const int K) {
    __m128 C_buffer[4][3]; // taking mR = 4 and nR = 12
    __m128 a_packFloat4;
    __m128 b0_packFloat4;
    __m128 b1_packFloat4;
    __m128 b2_packFloat4;

    for (int i = 0; i < 4; i++) {
        C_buffer[i][0] = _mm_loadu_ps(C + i * N);
        C_buffer[i][1] = _mm_loadu_ps(C + i * N + 4);
        C_buffer[i][2] = _mm_loadu_ps(C + i * N + 8);
    }

    for (int i = 0; i < K; i++) {
        b0_packFloat4 = _mm_loadu_ps(Bblock + i * N);
        b1_packFloat4 = _mm_loadu_ps(Bblock + i * N + 4);
        b2_packFloat4 = _mm_loadu_ps(Bblock + i * N + 8);

        a_packFloat4 = _mm_set1_ps(Ablock[i]);
        C_buffer[0][0] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b0_packFloat4), C_buffer[0][0]);
        C_buffer[0][1] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b1_packFloat4), C_buffer[0][1]);
        C_buffer[0][2] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b2_packFloat4), C_buffer[0][2]);

        a_packFloat4 = _mm_set1_ps(Ablock[i + K]);
        C_buffer[1][0] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b0_packFloat4), C_buffer[1][0]);
        C_buffer[1][1] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b1_packFloat4), C_buffer[1][1]);
        C_buffer[1][2] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b2_packFloat4), C_buffer[1][2]);

        a_packFloat4 = _mm_set1_ps(Ablock[i + K * 2]);
        C_buffer[2][0] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b0_packFloat4), C_buffer[2][0]);
        C_buffer[2][1] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b1_packFloat4), C_buffer[2][1]);
        C_buffer[2][2] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b2_packFloat4), C_buffer[2][2]);

        a_packFloat4 = _mm_set1_ps(Ablock[i + K * 3]);
        C_buffer[3][0] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b0_packFloat4), C_buffer[3][0]);
        C_buffer[3][1] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b1_packFloat4), C_buffer[3][1]);
        C_buffer[3][2] =
            _mm_add_ps(_mm_mul_ps(a_packFloat4, b2_packFloat4), C_buffer[3][2]);
    }

    for (int i = 0; i < 4; i++) {
        _mm_storeu_ps(C + i * N, C_buffer[i][0]);
        _mm_storeu_ps(C + i * N + 4, C_buffer[i][1]);
        _mm_storeu_ps(C + i * N + 8, C_buffer[i][2]);
    }

    return;
}

void kernel_matmul(matrix *A, matrix *B, matrix *out) {
    const int M = A->rows;
    const int N = B->cols;
    const int K = B->rows;

    float Abuffer[MR * K];
    float Bbuffer[K * NR];
    float Cbuffer[MR * NR];

    for (int i = 0; i < M; i += 4) {
        int m = ((M - i) < MR) ? (M - i) : MR;

        memcpy(Abuffer, A->data + i * K, sizeof(float) * K * m);
        memset(Abuffer + m * K, 0, sizeof(float) * K * (MR - m));

        /*
        printf("======== Abuffer %d =========\n", i);
        print_othermat(Abuffer, MR, K);
        */

        for (int j = 0; j < N; j += 12) {
            int n = ((N - j) < NR) ? (N - j) : NR;

            for (int nk = 0; nk < K; nk++) {
                memcpy(Bbuffer + nk * NR, B->data + nk * N + j,
                       sizeof(float) * n);
                memset(Bbuffer + nk * NR + n, 0, sizeof(float) * (NR - n));
            }

            int cm = 0;
            for (; cm < m; cm++) {
                memcpy(Cbuffer + cm * NR, out->data + i * N + j + cm * N,
                       sizeof(float) * n);
                memset(Cbuffer + cm * NR + n, 0, sizeof(float) * (NR - n));
            }
            memset(Cbuffer + cm * NR, 0, sizeof(float) * NR * (MR - cm));

            /*
            printf("======== Bbuffer %d =========\n", i);
            print_othermat(Bbuffer, K, NR);
            */

            kernel_4x12(Abuffer, Bbuffer, Cbuffer, MR, NR, K);

            // storing result in out
            for (int cm = 0; cm < m; cm++) {
                memcpy(out->data + i * N + j + cm * N, Cbuffer + cm * NR,
                       sizeof(float) * n);
            }
        }
    }

    return;
}

void increment_filler(matrix *mat) {
    int total_items = mat->total_data;
    float temp = 0;
    for (int i = 0; i < total_items; i++) {
        mat->data[i] = ++temp;
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

    /*
    printf("==================== A ====================\n");
    mat_print(a);
    printf("==================== B ====================\n");
    mat_print(b);
    */

    kernel_matmul(a, b, c);
    printf("Done with kernel matmul\n");
    naive_matmul(a, b, d);

    compare_matrix(c, d);

cleanup:
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(d);
    return 0;
}
