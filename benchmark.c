#include "benchmark.h"
#include "matrix.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#define MIN_SIZE 100
#define MAX_SIZE 2000
#define STEPS 15
#define NITER 20
#define WARMUP 5

uint64_t timer(void) {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

void single_bench(void matmul(matrix *, matrix *, matrix *)) {
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

    int size = MAX_SIZE;
    printf("For Size: %d\n", size);

    double FLOP = 2 * (double)size * size * size;

    long results[4];
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

    results[0] = size;
    results[1] = (long)(FLOP / max_time);
    results[2] = (long)(FLOP / min_time);
    results[3] = (long)(FLOP / avg_time);

    printf("Size: %d | min: %ld | max: %ld | avg: %ld\n", size, results[1],
           results[2], results[3]);

    free_matrix(mat_a);
    free_matrix(mat_b);
    free_matrix(mat_c);
    mat_a = mat_b = mat_c = NULL;

cleanup:
    free_matrix(mat_a);
    free_matrix(mat_b);
    free_matrix(mat_c);
    return;
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
