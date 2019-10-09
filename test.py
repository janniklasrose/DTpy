import DTpy

dt = DTpy.DiffusionTensor([1,0,0, 0,2,0, 0,0,3])
print("full tensor:")
print(dt.fullTensor)
print("eigenvalues:")
print(dt.eigenValues)
print("eigenvectors:")
print(dt.eigenVectors)
