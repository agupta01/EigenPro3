"""Utility functions for performing fast SVD."""
import torch.linalg as linalg
import torch
import math


def nystrom_kernel_svd(samples, kernel_fn, top_q):
    """Compute top eigensystem of kernel matrix using Nystrom method.
    Arguments:
        samples: data matrix of shape (n_sample, n_feature).
        kernel_fn: tensor function k(X, Y) that returns kernel matrix.
        top_q: top-q eigensystem.
    Returns:
        eigvals: top eigenvalues of shape (top_q).
        eigvecs: (rescaled) top eigenvectors of shape (n_sample, top_q).
    """
    # samples /= torch.max(samples).item()
    n_sample, _ = samples.shape
    kmat = kernel_fn(samples, samples)
    scaled_kmat = kmat / n_sample
    vals, vecs = torch.lobpcg(scaled_kmat, min(top_q + 1, n_sample // 3))
    # vals, vecs = linalg.eigh(scaled_kmat)
    # vals = vals[n_sample - top_q : n_sample - 1]
    # vecs = vecs[:, n_sample - top_q : n_sample - 1]
    # NOTE: torch.flip returns a copy of the tensor, not a view like np.flip
    # eigvals = torch.flip(vals, (0,))[:top_q]
    # eigvecs = torch.flip(vecs, (1,))[:, :top_q]
    beta = kmat.diag().max()

    return vals, vecs / math.sqrt(n_sample), beta
