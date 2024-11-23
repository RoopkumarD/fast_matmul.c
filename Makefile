naive_matmul: matrix.c naive_matmul.c
	gcc matrix.c naive_matmul.c -o naive_matmul.out -g

clean:
	rm naive_matmul.out
