naive_matmul: matrix.c naive_matmul.c
	gcc matrix.c naive_matmul.c -o naive_matmul.out -g

benchmark: matrix.c benchmark.c cmk_general.c
	gcc matrix.c benchmark.c cmk_general.c -o benchmark.out -g

clean:
	rm *.out
