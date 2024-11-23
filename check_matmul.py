import ast

import numpy as np

if __name__ == "__main__":
    with open("testing.txt", "r") as f:
        txt_file = f.read()

    txt_file = txt_file[:-1]
    for idx, line in enumerate(txt_file.split("\n")):
        temp = ast.literal_eval(line.replace("\0", ""))
        a, b, c = temp
        a = np.array(a).astype(np.float32)
        b = np.array(b).astype(np.float32)
        c = np.array(c).astype(np.float32)

        resl = a @ b
        print(f"For iter {idx} -> {np.allclose(resl, c, atol=1e-5)}")
