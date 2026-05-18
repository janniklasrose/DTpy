import numpy as np

from DTpy import DiffusionTensor


tensor = [
    [1, 0, 0],
    [0, 2, 0],
    [0.1, 0, 3]  # small perturbation
]


def fmt_number(value):
    return f"{float(value):.6g}"


def fmt_vector(vector):
    return f"({fmt_number(vector.x)}, {fmt_number(vector.y)}, {fmt_number(vector.z)})"


def fmt_numbers(values):
    return f"({', '.join(fmt_number(value) for value in values)})"


def fmt_vectors(vectors):
    return f"({', '.join(fmt_vector(vector) for vector in vectors)})"


def fmt_array(values):
    return np.array2string(values, precision=6, suppress_small=True)


def main():
    dt = DiffusionTensor(tensor)
    print("Tensor properties:")
    print(f"full tensor\n{fmt_array(dt.tensor)}")
    print(f"eigenvalues\n{fmt_numbers(dt.EigenValues)}")
    print(f"eigenvectors\n{fmt_vectors(dt.EigenVectors)}")
    print("Diffusion Tensor properties:")
    print(f"L1 = {fmt_number(dt.L1)}, L2 = {fmt_number(dt.L2)}, L3 = {fmt_number(dt.L3)}")
    print(f"E1 = {fmt_vector(dt.E1)}, E2 = {fmt_vector(dt.E2)}, E3 = {fmt_vector(dt.E3)}")
    print(f"MD = {fmt_number(dt.MD)}, FA = {fmt_number(dt.FA)}, Mo = {fmt_number(dt.Mo)}")


if __name__ == "__main__":
    main()
