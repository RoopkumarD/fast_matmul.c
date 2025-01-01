# Kernel Matrix Multiplication

Instead of performing matrix multiplication one vector at a time, we process subsets of matrices `A`
and `B` and perform matrix multiplication using SIMD and registers to achieve better performance.

Refer to:
- `123ck.c`
- `column_kernel.c` for column-order matrices
- `row_kernel.c` for row-order matrices

Also you can note that i am using `Cbuffer` cause i don't have mask load and store operation.

### Compilation Note

Compile with the `-O2` flag to enable efficient usage of the XMM registers. Without this
optimization, the compiler may copy data to aligned memory on the stack, causing unnecessary
overhead. Examine the assembly code for non-`-O2` compiled binaries to observe this behavior.

```c
void kernel_12x3(float *Ablock, float *Bblock, float *C, const int K) {
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

    return;
}

void kernel_matmul(matrix *A, matrix *B, matrix *out) {
    int M = A->rows;
    int N = B->cols;
    int K = B->rows;

    float Abuffer[MR * K];
    float Bbuffer[K * NR];
    float Cbuffer[MR * NR];

    for (int i = 0; i < M; i += MR) {
        int m = ((M - i) < MR) ? (M - i) : MR;

        // also notice that i am doing copying of for loop memory
        // outside. refer row_kernel.c where i am copying Bbuffer
        // outside cause copying inside is very costly as it takes
        // M/MR * N/NR loop for the for loop
        // that's why copying outside will only require M/MR
        // inside due to continous memory i didn't have to use
        // for loop and did directly memcpy

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

            kernel_12x3(Abuffer, Bbuffer, Cbuffer, K);

            // storing result in out
            for (int cn = 0; cn < n; cn++) {
                memcpy(out->data + j * M + i + cn * M, Cbuffer + cn * MR,
                       sizeof(float) * m);
            }
        }
    }
    return;
}
```

Here, `MR = 12` and `NR = 3` gave me good GFLOPS compared to other values for MR and NR. The reason
for choosing `NR = 3` over `NR = 4` is that using `NR = 4` results in register spills. I initially
thought I could use all the XMM registers, but the assembly code showed otherwise. Using `NR = 3`
reduces register use and avoids saving extra values in aligned memory, thus saving time.

### Todo

1. Investigate why the code uses only one XMM register instead of utilizing all available registers
   for kernel matrix multiplication. Analyze the compiled assembly code to identify where registers
   are used and rewrite the code to utilize all XMM registers.

Another attempt was to conditionally use either `A` or `Abuffer` by first looping through multiples
of `MR` and then handling the rest in another loop, as shown below:

```c
int i = 0;
for (; i + MR < M; i += MR) {
    // ...
}
// Handle the rest of the A matrix
```

However, this didn’t improve performance, even though I believe (without proof) that memory is a
blocking factor in matrix multiplication. Refer to `condition_ck.c`.

### Todo

2. Understand why the above approach didn’t speed up matrix multiplication.
3. Examine how much memory copying is blocking matrix multiplication.

### Batched Matrix Multiplication

This approach seems slow, which makes sense as I am doing extra work for small matrices. I suspect
that my tiling strategy might be incorrect (needs more research as I didn’t think it through, just
copied the tutorial’s tiling). Additionally, I think memory copying in loops is consuming the
majority of the time compared to the internal memory copying done only once in the normal approach
(though I haven’t checked real values, this is based on intuition). Refer to `batched_ck.c`.

### Todo

4. Understand tiling and caching blocks and use them for `kernel_matmul`.

### OpenMP Parallelization

Using OpenMP as seen in `batching.c` gave me 24 GFLOPS.

### Todo

5. Understand why the tutorial used aligned memory and why aligning memory to 32 bytes makes matrix
   multiplication slightly faster.
