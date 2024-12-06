naive_matmul: matrix.c naive_matmul.c
	gcc matrix.c naive_matmul.c -o naive_matmul.out -g

benchmark: matrix.c benchmark.c
	gcc matrix.c benchmark.c -o benchmark.out -g

clean:
	rm *.out
