for i in range(1, 16):
    for j in range(1, 16):
        if (i * j) % 4 == 0 and i % 4 == 0:
            k = (i * j) // 4
            m = i // 4
            if m != 0 and k + m + 1 <= 16:
                print(i, j)
