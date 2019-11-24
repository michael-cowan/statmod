import numpy as np


def pca(x, tranform=None):
    """
    PCA transformation using numpy
    - calculate covariance matrix
    - find eigenvectors (transformation matrix) of
      covariance matrix
    - features (dot) eigenvectors = tranformed features

    Args:
    x (np.ndarray): features

    KArgs:
    transform (np.ndarray): if given, used to transform features
                            - transforms new features from different PCA
                            (Default: None)

    Returns (dict):
    pcs: principal components (transformed features)
    transform: transformation matrix to convert other features into PCs
    explained_variance: amount of explained variance by each PC
    explained_variance_ratio: percentage of explained variance by each PC
    """

    # if transform supplied, apply it to pos and return
    if isinstance(tranform, np.ndarray):
        return np.dot(x, tranform)

    var = np.var(x, axis=0)

    # center data
    x -= x.mean(axis=0)

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

    return dict(pcs=x_pca, transform=evecs, explained_variance=evals,
                explained_variance_ratio=evals / evals.sum())


if __name__ == '__main__':
    x = np.random.random((1000, 100))
    res = pca(x)
    print(res['explained_variance_ratio'][:5])
