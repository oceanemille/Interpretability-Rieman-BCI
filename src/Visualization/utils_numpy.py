import numpy as np
from sklearn.decomposition import PCA
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import log_map_riemann, exp_map_riemann
from pyriemann.utils.tangentspace import upper, unupper


def pga(covs, random_state=None):
    mean = mean_riemann(covs)
    # First whiten the data and send to tangent space at identity.
    # log(mean^{-1/2} @ covs @ mean^{-1/2})
    all_u = log_map_riemann(covs, mean, False)

    # Then flatten the upper triangular part of the tangent vectors to get a Euclidean representation.
    flat_covs = upper(all_u)

    # Perform PCA in the Euclidean space.
    pca = PCA(n_components=3, random_state=random_state)
    covs_embedded = pca.fit_transform(flat_covs)
    # PCA is rescaled so that PC1 has standard deviation 1e-4
    covs_embedded = covs_embedded / np.std(covs_embedded[:, 0]) * 1e-4

    # Finally, map the embedded points back to the manifold using the exponential map at the identity.
    return exp_map_riemann(unupper(covs_embedded), np.eye(2))


def pairwise_airm_dist_2x2(X, Y=None):
    if Y is None:
        Y = X

    aX = X[:, 0, 0]
    bX = X[:, 0, 1]
    cX = X[:, 1, 1]
    aY = Y[:, 0, 0]
    bY = Y[:, 0, 1]
    cY = Y[:, 1, 1]
    detX = aX * cX - bX * bX
    detY = aY * cY - bY * bY

    trace = (np.outer(cX, aY) - 2 * np.outer(bX, bY) + np.outer(aX, cY)) / detX[:, None]
    det = np.outer(1 / detX, detY)

    delta = trace**2 - 4 * det
    delta = np.maximum(delta, 0.0)  # Avoid negative values due to numerical issues
    sqrt_delta = np.sqrt(delta)
    lmb_plus = 0.5 * (trace + sqrt_delta)
    lmb_minus = 0.5 * (trace - sqrt_delta)
    return np.log(lmb_plus) ** 2 + np.log(lmb_minus) ** 2


def eig_dec_2x2(X):
    a = X[:, 0, 0]
    b = X[:, 1, 0]
    c = X[:, 1, 1]

    amc = a - c
    delta = np.sqrt(amc**2 + 4 * b * b)

    lmb_p = 0.5 * ((a + c) + delta)
    lmb_m = 0.5 * ((a + c) - delta)

    # Norms  shape (n,)
    norm_p = np.sqrt(b**2 + (a - lmb_p) ** 2)
    norm_m = np.sqrt(b**2 + (a - lmb_m) ** 2)

    # Eigenvectors  shape (n, 2)
    v_p = np.stack([-b, a - lmb_p], axis=1) / norm_p[:, None]
    v_m = np.stack([-b, a - lmb_m], axis=1) / norm_m[:, None]

    # Eigenvalues: (n, 2)
    # Eigenvector matrix: (n, 2, 2) — columns are eigenvectors
    lambdas = np.stack([lmb_m, lmb_p], axis=1)
    vectors = np.empty((X.shape[0], 2, 2))
    vectors[:, :, 0] = v_m
    vectors[:, :, 1] = v_p

    return lambdas, vectors


def reconstruct_from_eig(lmb, P):
    v1 = P[:, :, 0]
    v2 = P[:, :, 1]

    return lmb[:, 0][:, None, None] * (v1[:, :, None] * v1[:, None, :]) + lmb[:, 1][
        :, None, None
    ] * (v2[:, :, None] * v2[:, None, :])


def sqrt_inv_sqrt_2x2(X):
    lmb, P = eig_dec_2x2(X)
    sqrt_lmb = np.sqrt(lmb)
    inv_sqrt_lmb = 1.0 / sqrt_lmb
    return reconstruct_from_eig(sqrt_lmb, P), reconstruct_from_eig(inv_sqrt_lmb, P)


def log_2x2(X):
    lmb, P = eig_dec_2x2(X)
    log_lmb = np.log(lmb)
    return reconstruct_from_eig(log_lmb, P)


def log_riemannian_2x2(X, Cref):
    Cref_sqrt, Cref_inv_sqrt = sqrt_inv_sqrt_2x2(Cref[None])
    log_factor = log_2x2(Cref_inv_sqrt @ X @ Cref_inv_sqrt)
    return Cref_sqrt @ log_factor @ Cref_sqrt


def multitransp(A):
    """Vectorized matrix transpose.

    ``A`` is assumed to be an array containing ``M`` matrices, each of which
    has dimension ``N x P``.
    That is, ``A`` is an ``M x N x P`` array. Multitransp then returns an array
    containing the ``M`` matrix transposes of the matrices in ``A``, each of
    which will be ``P x N``.
    """
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))


def multisym(A):
    """Vectorized matrix symmetrization.

    Given an array ``A`` of matrices (represented as an array of shape ``(k, n,
    n)``), returns a version of ``A`` with each matrix symmetrized, i.e.,
    every matrix ``A[i]`` satisfies ``A[i] == A[i].T``.
    """
    return 0.5 * (A + multitransp(A))


def retraction(point, tangent_vector):
    p_inv_tv = np.linalg.solve(point, tangent_vector)
    return multisym(point + tangent_vector + tangent_vector @ p_inv_tv / 2)


def norm_SPD(point, tangent_vector):
    p_inv_tv = np.linalg.solve(point, tangent_vector)
    return np.sqrt(
        np.tensordot(p_inv_tv, multitransp(p_inv_tv), axes=tangent_vector.ndim)
    )
    
    
    
    
    
    
