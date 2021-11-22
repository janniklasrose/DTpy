from DTpy import DiffusionTensor


tensor = [
    [1, 0, 0],
    [0, 2, 0],
    [0.1, 0, 3]  # small perturbation
]

dt = DiffusionTensor(tensor)

print("Tensor properties:")
print("full tensor\n", dt.tensor)
print("eigenvalues\n", dt.EigenValues)
print("eigenvectors\n", dt.EigenVectors)
print("Diffusion Tensor properties:")
print(f"L1 = {dt.L1}, L2 = {dt.L2}, L3 = {dt.L3}")
print(f"E1 = {dt.E1}, E2 = {dt.E2}, E3 = {dt.E3}")
print(f"MD = {dt.MD}, FA = {dt.FA}, Mo = {dt.Mo}")
