#include "matrix.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#define MIN_SIZE 100
#define MAX_SIZE 1000
#define STEPS 10
#define NITER 20
#define WARMUP 5

int main(void) {
    srand(time(NULL));

    struct winsize w;

    // Get terminal size
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        perror("ioctl");
        return 1;
    }

    int width_mul = w.ws_col / NITER;
    char loader[width_mul + 1];
    memset(loader, '#', width_mul);
    loader[width_mul] = '\0';

    matrix *mat_a = NULL;
    matrix *mat_b = NULL;
    matrix *mat_c = NULL;

    int arrsizes[STEPS + 1];
    for (int i = 0; i <= STEPS; i++) {
        double multiplier = (double)(MAX_SIZE - MIN_SIZE) / STEPS;
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
        cache_matmul(mat_a, mat_b, mat_c);
    }
    printf("WARMUP END\n");
    free_matrix(mat_a);
    free_matrix(mat_b);
    free_matrix(mat_c);
    mat_a = mat_b = mat_c = NULL;

    for (int i = 0; i <= STEPS; i++) {
        int size = arrsizes[i];
        printf("For Size: %d\n", size);

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

        clock_t min_time = LONG_MAX;
        clock_t max_time = 0;
        clock_t avg_time = 0;

        for (int j = 0; j < NITER; j++) {
            printf("%s", loader);
            fflush(stdout);

            fill_random(mat_a);
            fill_random(mat_b);
            memset(mat_c->data, 0, mat_c->total_data * sizeof(float));

            clock_t start = clock();
            cache_matmul(mat_a, mat_b, mat_c);
            clock_t end = clock();

            clock_t time_elasped = end - start;
            avg_time += time_elasped;
            min_time = (time_elasped < min_time) ? time_elasped : min_time;
            max_time = (time_elasped > max_time) ? time_elasped : max_time;
        }
        printf("%s\n", loader);
        avg_time = avg_time / NITER;

        results[i][0] = size;
        results[i][1] = min_time;
        results[i][2] = max_time;
        results[i][3] = avg_time;

        free_matrix(mat_a);
        free_matrix(mat_b);
        free_matrix(mat_c);
        mat_a = mat_b = mat_c = NULL;
    }

    // saving the results to file
    char *filename = "cache_benchmark_results.txt";
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
    return 0;
}
