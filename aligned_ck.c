// Since i have SSE instruction set -> thus 16 XMM registers and no FMA
// instructions So i will do the both FMA in two instruction
//
// Column Major Order

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <xmmintrin.h>

#include <limits.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

typedef struct {
    int total_data;
    int rows;
    int cols;
    float *data;
} matrix;

#define DO_BENCH 1

#define MR 12
#define NR 3

#define MIN_SIZE 100
#define MAX_SIZE 2000
#define STEPS 15
#define NITER 20
#define WARMUP 5

#define MEM_ALIGN 32

matrix *mat_create(int rows, int cols) {
    int total_data = rows * cols;
    float *data = _mm_malloc(total_data * sizeof(float), MEM_ALIGN);
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
    _mm_free(mat->data);
    free(mat);
    return;
}

void fill_random(matrix *mat) {
    size_t total_data = mat->total_data;
    for (size_t i = 0; i < total_data; i++) {
        mat->data[i] = ((double)rand() / (RAND_MAX + 1.0)) - 0.5;
    }
    return;
}

uint64_t timer(void) {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

void benchmark(void matmul(matrix *, matrix *, matrix *), char *filename) {
    struct winsize w;

    // Get terminal size
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        perror("ioctl");
        return;
    }

    int width_mul = w.ws_col / NITER;
    char loader[width_mul + 1];
    memset(loader, '#', width_mul);
    loader[width_mul] = '\0';

    matrix *mat_a = NULL;
    matrix *mat_b = NULL;
    matrix *mat_c = NULL;

    int arrsizes[STEPS + 1];
    double multiplier = (double)(MAX_SIZE - MIN_SIZE) / STEPS;
    for (int i = 0; i <= STEPS; i++) {
        arrsizes[i] = multiplier * i + MIN_SIZE;
    }

    long results[STEPS + 1][4];

    // warmup
    mat_a = mat_create(MAX_SIZE, MAX_SIZE);
    if (mat_a == NULL) {
        goto cleanup;
    }

    mat_b = mat_create(MAX_SIZE, MAX_SIZE);
    if (mat_b == NULL) {
        goto cleanup;
    }

    mat_c = mat_create(MAX_SIZE, MAX_SIZE);
    if (mat_c == NULL) {
        goto cleanup;
    }
    printf("WARMUP START\n");
    for (int i = 0; i < WARMUP; i++) {
        fill_random(mat_a);
        fill_random(mat_b);
        memset(mat_c->data, 0, mat_c->total_data * sizeof(float));
        matmul(mat_a, mat_b, mat_c);
        printf("%d ", i);
        fflush(stdout);
    }
    printf("\nWARMUP END\n");
    free_matrix(mat_a);
    free_matrix(mat_b);
    free_matrix(mat_c);
    mat_a = mat_b = mat_c = NULL;

    for (int i = 0; i <= STEPS; i++) {
        int size = arrsizes[i];
        printf("For Size: %d\n", size);

        double FLOP = 2 * (double)size * size * size;

        mat_a = mat_create(size, size);
        if (mat_a == NULL) {
            goto cleanup;
        }

        mat_b = mat_create(size, size);
        if (mat_b == NULL) {
            goto cleanup;
        }

        mat_c = mat_create(size, size);
        if (mat_c == NULL) {
            goto cleanup;
        }

        double min_time = INFINITY;
        double max_time = 0;
        double avg_time = 0;

        for (int j = 0; j < NITER; j++) {
            printf("%s", loader);
            fflush(stdout);

            fill_random(mat_a);
            fill_random(mat_b);
            memset(mat_c->data, 0, mat_c->total_data * sizeof(float));

            uint64_t start = timer();
            matmul(mat_a, mat_b, mat_c);
            uint64_t end = timer();

            double time_elasped = (end - start) * 1e-9;
            avg_time += time_elasped;
            min_time = (time_elasped < min_time) ? time_elasped : min_time;
            max_time = (time_elasped > max_time) ? time_elasped : max_time;
        }
        printf("%s\n", loader);
        avg_time = avg_time / NITER;

        results[i][0] = size;
        results[i][1] = (long)(FLOP / max_time);
        results[i][2] = (long)(FLOP / min_time);
        results[i][3] = (long)(FLOP / avg_time);

        printf("Size: %d | min: %ld | max: %ld | avg: %ld\n", size,
               results[i][1], results[i][2], results[i][3]);

        free_matrix(mat_a);
        free_matrix(mat_b);
        free_matrix(mat_c);
        mat_a = mat_b = mat_c = NULL;
    }

    // saving the results to file
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Couldn't open file");
        goto cleanup;
    }

    for (int i = 0; i <= STEPS; i++) {
        fprintf(fp, "%ld %ld %ld %ld\n", results[i][0], results[i][1],
                results[i][2], results[i][3]);
    }

    printf("WRITTEN TO FILE: %s\n", filename);
    fclose(fp);

cleanup:
    free_matrix(mat_a);
    free_matrix(mat_b);
    free_matrix(mat_c);
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

    /*
    __m128 C_buffer30 = _mm_loadu_ps(C + 3 * MR);
    __m128 C_buffer31 = _mm_loadu_ps(C + 3 * MR + 4);
    __m128 C_buffer32 = _mm_loadu_ps(C + 3 * MR + 8);
    */

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

        /*
        b_packFloat4 = _mm_set1_ps(Bblock[p + K * 3]);
        C_buffer30 =
            _mm_add_ps(_mm_mul_ps(a0_packFloat4, b_packFloat4), C_buffer30);
        C_buffer31 =
            _mm_add_ps(_mm_mul_ps(a1_packFloat4, b_packFloat4), C_buffer31);
        C_buffer32 =
            _mm_add_ps(_mm_mul_ps(a2_packFloat4, b_packFloat4), C_buffer32);
            */
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

    /*
    _mm_storeu_ps(C + 3 * MR, C_buffer30);
    _mm_storeu_ps(C + 3 * MR + 4, C_buffer31);
    _mm_storeu_ps(C + 3 * MR + 8, C_buffer32);
    */

    return;
}

void kernel_matmul(matrix *A, matrix *B, matrix *out) {
    int M = A->rows;
    int N = B->cols;
    int K = B->rows;

    float Abuffer[MR * K] __attribute__((aligned(MEM_ALIGN)));
    float Bbuffer[K * NR] __attribute__((aligned(MEM_ALIGN)));
    float Cbuffer[MR * NR] __attribute__((aligned(MEM_ALIGN)));

    for (int i = 0; i < M; i += MR) {
        int m = ((M - i) < MR) ? (M - i) : MR;

        // rather than memset(Ab, 0, (m < MR) * ...); this
        // as this is always fast
        memset(Abuffer, 0, sizeof(float) * MR * K);
        for (int mk = 0; mk < K; mk++) {
            memcpy(Abuffer + mk * MR, A->data + mk * M + i, sizeof(float) * m);
        }

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
    sprintf(filename, "benchmarks/%dx%d_column_aligned_%d.txt", MR, NR,
            MEM_ALIGN);
    benchmark(kernel_matmul, filename);
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
