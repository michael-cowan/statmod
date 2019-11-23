import numpy as np


def pca(x, return_transform=False, tranform=None):
    """
    PCA transformation using numpy
    - calculate covariance matrix
    - find eigenvectors (transformation matrix) of
      covariance matrix
    - features (dot) eigenvectors = tranformed features

    Args:
    x (np.ndarray): features

    KArgs:
    return_transform (bool): if True, eigenvectors of covariance matrix is
                             returned with the transformed features
                             (Default: False)
    transform (np.ndarray): if given, used to transform features
                            - transforms new features from different PCA
                            (Default: None)
    """
    # if transform supplied, apply it to pos and return
    if isinstance(tranform, np.ndarray):
        return np.dot(x, tranform)

    var = np.var(x, axis=0)

    # covariance = (x.T * x) / (n - 1)
    # do not need to subtract off mean since data is centered
    co = np.dot(x.T, x) / (len(x) - 1)

    # find eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(co)

    # sort evals
    indices = evals.argsort()[::-1]
    evals = evals[indices]
    evecs = evecs[:, indices]

    # eigenvectors = transformation matrix
    x_pca = np.dot(x, evecs)

    # return evecs if return_transform
    if return_transform:
        return x_pca, evecs
    else:
        return x_pca
    return x_pca, evecs if return_transform else x_pca


if __name__ == '__main__':
    x = np.random.random((100, 10))
    pca_x, trans = pca(x, return_transform=True)
