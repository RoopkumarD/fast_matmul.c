#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

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

void fill_random(matrix *mat) {
    size_t total_data = mat->total_data;
    for (size_t i = 0; i < total_data; i++) {
        mat->data[i] = ((double)rand() / (RAND_MAX + 1.0)) - 0.5;
    }
    return;
}

void mat_print(matrix *mat) {
    int cols = mat->cols;
    int rows = mat->rows;
    printf("[");
    for (int i = 0; i < rows; i++) {
        printf("%s", (i != 0) ? " [" : "[");
        for (int j = 0; j < cols; j++) {
            printf("%f", mat->data[i * cols + j]);
            if (j != cols - 1) {
                printf(", ");
            }
        }
        printf("%s", (i != rows - 1) ? "],\n" : "]");
    }
    printf("]\n");
    return;
}

size_t mat_text(matrix *mat, char *string, size_t idx) {
    int cols = mat->cols;
    int rows = mat->rows;
    string[idx++] = '[';
    for (int i = 0; i < rows; i++) {
        string[idx++] = '[';
        for (int j = 0; j < cols; j++) {
            idx += snprintf(string + idx, 32, "%f", mat->data[i * cols + j]);
            if (j != cols - 1) {
                string[idx++] = ',';
            }
        }
        string[idx++] = ']';
        if (i != rows - 1) {
            string[idx++] = ',';
        }
    }
    string[idx++] = ']';
    return idx;
}

void naive_matmul(matrix *a, matrix *b, matrix *out) {
    int cols1 = a->cols;
    int rows1 = a->rows;
    int cols2 = b->cols;
    int rows2 = b->rows;

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                out->data[i * cols2 + j] +=
                    a->data[i * cols1 + k] * b->data[k * cols2 + j];
            }
        }
    }

    return;
}

/*
 * Based on using cache locality, where i take one row
 * from left mat and other row from right mat.
 *
 * Refer README for explanation or use any 2x2 matrix example
 */
void cache_matmul(matrix *a, matrix *b, matrix *out) {
    int cols1 = a->cols;
    int rows1 = a->rows;
    int cols2 = b->cols;
    int rows2 = b->rows;

    // O[i][k] = C[i][j] * S[j][k]
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols1; j++) {
            for (int k = 0; k < cols2; k++) {
                out->data[i * cols2 + k] +=
                    a->data[i * cols1 + j] * b->data[j * cols2 + k];
            }
        }
    }

    return;
}
