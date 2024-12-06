import numpy as np

np.random.seed(42)

a = np.random.randint(100, size=(3, 4))
b = np.random.randint(100, size=(4, 2))
print(a)
print(b)

c = a @ b

m = np.zeros((3, 2))

for i in range(3):
    for j in range(4):
        for k in range(2):
            m[i][k] += a[i][j] * b[j][k]
            print(
                f"{a[i][j]}(a[{i}][{j}]) * {b[j][k]}(b[{i}][{j}]) = {m[i][k]}(m[{i}][{j}])"
            )
            print(m)


print(c)
print(m)
