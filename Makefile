# added -O2 optimisation flag because without it, the loadu operation always
# loads value to xmm0 registers always. So it only using xmm0 registers leaving
# every other xmm registers.
# tried with -O1 but still same result as without it
# that's why without it i wasn't getting GFLOPS but after adding it
# it gave me 14FLOPS MAX as it is using all xmm register
#
# Found about this when i looked at compiled assembly code

naive_matmul: matrix.c naive_matmul.c
	gcc matrix.c naive_matmul.c -o naive_matmul.out -g -march=native -O2

column_benchmark: matrix.c benchmark.c column_kernel.c
	gcc matrix.c benchmark.c column_kernel.c -o column_kernel.out -g -march=native -O2

row_benchmark: matrix.c benchmark.c row_kernel.c
	gcc matrix.c benchmark.c row_kernel.c -o row_kernel.out -g -march=native -O2

clean:
	rm *.out
