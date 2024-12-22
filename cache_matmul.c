#include "matrix.h"

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

int main(void) {
    // test or benchmark it
    return 0;
}
