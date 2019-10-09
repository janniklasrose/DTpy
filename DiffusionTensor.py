import numpy as np

def eigenDecomposition(tensor):
    """
    Returns the eigenvalues and eigenvectors of the tensor

    The output is sorted by descending magnitude of the eigenvalues, such that
    the primary eigenvector E1 has the largest corresponding eigenvalue L1.

    Input:
        tensor - any 2D array that np.linalg.eig can process
    Output:
        eVal - eigenvalues in descending order
        eVec - eigenvectors corresponding to the eigenvalues

    See <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix>
    """

    # use eig for simplicity, accuracy, robustness
    D, V = np.linalg.eig(tensor)  # eigenvalues, eigenvectors

    # sort (documentation says .eig does not guarantee that D is ordered!)
    idx = np.argsort(D)[::-1]  # reverse the output to get descending order

    # output
    eVal = D[idx]
    eVec = V[:, idx]
    return (eVal, eVec)

class DiffusionTensor:

    def __init__(self, tensor):
        """
        Construct a DiffusionTensor object from the raw tensor data

        Input:
            tensor - 2D (3x3) or 1D (6 elements) array
        """

        # process
        tensor = np.array(tensor)  # change to numpy if not already
        tensor = np.reshape(tensor, (3,3))  # in case 1D was given

        # calculate
        eVal, eVec = eigenDecomposition(tensor)

        # store
        self.fullTensor = tensor
        self.eigenValues = eVal
        self.eigenVectors = eVec
