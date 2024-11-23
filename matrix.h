#include <stdlib.h>

#ifndef MATRIX_H
#define MATRIX_H 1

typedef struct {
    int total_data;
    int rows;
    int cols;
    float *data;
} matrix;

matrix *mat_create(int rows, int cols);
void free_matrix(matrix *mat);
void fill_random(matrix *mat);
void mat_print(matrix *mat);
size_t mat_text(matrix *mat, char *string, size_t idx);

void naive_matmul(matrix *a, matrix *b, matrix *out);

#endif // !MATRIX_H
