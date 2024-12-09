// Row Major Order
//
// Same as Column Major Order but the only difference is we are
// taking one element to broadcast from A matrix than B
// Will result in same intermediate matrix as that of Column Order
// because we are matrix multiplying one vector to other thus one element
// at once looking from both direction

#include "matrix.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

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

    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 12) {
            kernel_4x12(A->data + i * K, B->data + j, out->data + i * N + j, M,
                        N, K);
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

    const int M = 4 * 20;
    const int N = 12 * 10;
    const int K = 1 * 100;

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
    naive_matmul(a, b, d);

    compare_matrix(c, d);

cleanup:
    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(d);
    return 0;
}
