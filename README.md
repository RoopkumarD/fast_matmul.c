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

## Algorithms

Refer under notes directory about each algorithm i wrote here.

## Todo's

- `notes/cache_matmul.md` delve into the reason why it is fast, let's not use vague terms and really
  understand why and how it works.
