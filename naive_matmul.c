#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SIZE 5
#define NITER 10

int main2(void) {
    srand(time(NULL));

    matrix *mat_a = NULL;
    matrix *mat_b = NULL;
    matrix *mat_c = NULL;

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

    for (int i = 0; i < NITER; i++) {
        fill_random(mat_a);
        fill_random(mat_b);
        naive_matmul(mat_a, mat_b, mat_c);
        memset(mat_a->data, 0, mat_a->total_data * sizeof(float));
        memset(mat_b->data, 0, mat_b->total_data * sizeof(float));
        memset(mat_c->data, 0, mat_c->total_data * sizeof(float));
    }

cleanup:
    free_matrix(mat_a);
    free_matrix(mat_b);
    free_matrix(mat_c);
    return 0;
}

int main(void) {
    srand(time(NULL));

    matrix *mat_a = NULL;
    matrix *mat_b = NULL;
    matrix *mat_c = NULL;

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

    char *mat_string = calloc(2 * 1024, sizeof(char));
    size_t idx = 0;

    // testing if matrix multiplication is correct or not
    FILE *fp = fopen("testing.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Couldn't Open the file\n");
        free(mat_string);
        goto cleanup;
    }

    for (int i = 0; i < NITER; i++) {
        fill_random(mat_a);
        fill_random(mat_b);
        naive_matmul(mat_a, mat_b, mat_c);
        idx += mat_text(mat_a, mat_string, idx);
        mat_string[idx++] = ',';
        idx += mat_text(mat_b, mat_string, idx);
        mat_string[idx++] = ',';
        idx += mat_text(mat_c, mat_string, idx);
        mat_string[idx++] = '\n';
        fwrite(mat_string, sizeof(char), idx, fp);
        memset(mat_string, 0, idx * sizeof(char));
        idx = 0;
        memset(mat_a->data, 0, mat_a->total_data * sizeof(float));
        memset(mat_b->data, 0, mat_b->total_data * sizeof(float));
        memset(mat_c->data, 0, mat_c->total_data * sizeof(float));
    }

    free(mat_string);
    fclose(fp);

cleanup:
    free_matrix(mat_a);
    free_matrix(mat_b);
    free_matrix(mat_c);
    return 0;
}
