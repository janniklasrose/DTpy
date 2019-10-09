import DTpy

tensor = [[1,0,0], [0,2,0], [0.1,0,3]]  # small perturbation

dt = DTpy.DiffusionTensor(tensor)

print("full tensor:")
print(dt.fullTensor)
print("eigenvalues:")
print(dt.eigenValues)
print("eigenvectors:")
print(dt.eigenVectors)
print("E1, E2, E3:")
print(dt.E1)
print(dt.E2)
print(dt.E3)
print("L1, L2, L3:")
print(dt.L1)
print(dt.L2)
print(dt.L3)
