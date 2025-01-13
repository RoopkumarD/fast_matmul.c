# Fast Matrix Multiplication (MatMul)

This project implements a faster matrix multiplication algorithm inspired by [this
article](https://github.com/salykova/matmul.c). The goal is to optimize matrix multiplication
tailored for my hardware.

## System Specifications

The benchmarks were conducted on the following hardware and software:

- **CPU**: x86_64, 2 Cores, 2 Threads  
  - Base Frequency: 1.6 GHz  
  - Turbo Frequency: 2.5 GHz  
  - Cache:  
    - L1: 64 KiB (per core)  
    - L2: 512 KiB (per core)  
    - L3: 3 MiB (shared)  
- **RAM**: 4 GB DDR3 (1333 MT/s)  
- **OS**: Linux 5.15.0-126-generic  
- **Compiler**: Ubuntu Clang 14.0.0  
- **Library**: NumPy 1.26.1  
- **Flags**: mmx, sse, sse2, ssse3, sse4_1, sse4_2, pclmulqdq

## Algorithms

Refer under notes directory about each algorithm i wrote here.

## Todo's

- important future todo: seriously able to read and interpret the compiled assembly and have 100%
  confidence about what is occuring at each step

- `notes/cache_matmul.md` delve into the reason why it is fast, let's not use vague terms and really
  understand why and how it works.

- `notes/kernel_matmul.md` delves into small block matrix multiplication with all other stuff to
  make matrix multiplication fast. There are many things i didn't understand and i wrote each of
  them in Todo's block, so solve them future me.

- Lastly above code didn't beat numpy implementation which had 35 - 36 GFLOPS whereas i had 24 GFLOP
  with openmp parallisation. Let's improve with those todo's and also refer to future direction by
  author conclusion note.  
  Also checkout other implementation like BLAS, etc and make our matrix multiplication faster than
  them.

- another direction is of taking fast matrix transpose of B, and then doing above stuff but still is it
  worth it. see if doing so will result in faster matrix multiplication.

- another [implementation](https://gist.github.com/snowclipsed/24aebbdd51218f4e17ad428630cc91d6) to take
  reference of which outperforms implementation of numpy and such.
