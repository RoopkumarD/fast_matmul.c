// Since i have SSE instruction set -> thus 16 XMM registers and no FMA
// instructions So i will do the both FMA in two instruction
//
// Column Major Order

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

typedef struct {
    int total_data;
    int rows;
    int cols;
    float *data;
} matrix;

matrix *mat_create(int rows, int cols) {
    int total_data = rows * cols;
    float *data = calloc(total_data, sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Couldn't allocate memory for array of size -> %d\n",
                total_data);
        return NULL;
    }
    matrix *mat = malloc(sizeof(matrix));
    if (mat == NULL) {
        fprintf(stderr, "Couldn't allocate memory for mat struct\n");
        free(data);
        return NULL;
    }
    mat->rows = rows;
    mat->cols = cols;
    mat->total_data = rows * cols;
    mat->data = data;
    return mat;
}

void free_matrix(matrix *mat) {
    if (mat == NULL) {
        return;
    }
    free(mat->data);
    free(mat);
    return;
}

void fill_random(matrix *mat, int high) {
    size_t total_data = mat->total_data;
    for (size_t i = 0; i < total_data; i++) {
        mat->data[i] = ((double)rand() / (RAND_MAX + 1.0)) - 0.5;
    }
    return;
}

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

void matrix_print(matrix *mat) {
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

void matmul_naive(matrix *a, matrix *b, matrix *c) {
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

void kernel_matmul(matrix *a, matrix *b, matrix *c) {
    int rows = a->rows;
    int common = a->cols;
    int cols = b->cols;

    for (int i = 0; i < rows; i += 12) {
        for (int j = 0; j < cols; j += 4) {
            kernel_12x4(a->data + i, b->data + j * common,
                        c->data + j * rows + i, rows, cols, common);
        }
    }
    return;
}

uint64_t timer(void) {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main(void) {
    srand(time(NULL));

    const int M = 12 * 1;
    const int N = 4 * 1;
    const int K = 1 * 1;

    matrix *a = mat_create(M, K);
    matrix *b = mat_create(K, N);
    matrix *c = mat_create(M, N);
    matrix *d = mat_create(M, N);

    fill_random(a, 100);
    fill_random(b, 100);
    memset(c->data, 0, c->total_data * sizeof(float));
    memset(d->data, 0, d->total_data * sizeof(float));

    kernel_matmul(a, b, c);
    matmul_naive(a, b, d);

    matrix_print(c);

    compare_mats(c->data, d->data, M, N);

    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_matrix(d);
    return 0;
}
