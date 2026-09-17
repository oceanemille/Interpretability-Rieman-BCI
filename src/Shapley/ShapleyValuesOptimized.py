import math
import numpy as np
from joblib import Parallel, delayed
import time
from src.Shapley.shapley_spd import proj_on_spd


import numpy as np


def _inv_sqrt_spd(M):
    eigvals, eigvecs = np.linalg.eigh(M)

    eigvals = np.maximum(eigvals, 1e-12)

    return (
        eigvecs
        * (1.0 / np.sqrt(eigvals))[None, :]
    ) @ eigvecs.T


def _riemann_distance_batch(X, M_inv_sqrt):

    # M^{-1/2} X M^{-1/2}
    whitened = (
        M_inv_sqrt[None, :, :]
        @ X
        @ M_inv_sqrt[None, :, :]
    )

    # Eigenvalues for every matrix in the batch
    eigvals = np.linalg.eigvalsh(whitened)

    eigvals = np.maximum(eigvals, 1e-12)

    distances = np.sqrt(
        np.sum(np.log(eigvals) ** 2, axis=1)
    )

    return distances

def mdm_predict_proba_fast(
    X,
    covmeans,
    target_class_idx=0,
):

    M0_inv_sqrt = _inv_sqrt_spd(covmeans[0])
    M1_inv_sqrt = _inv_sqrt_spd(covmeans[1])

    d0 = _riemann_distance_batch(
        X,
        M0_inv_sqrt,
    )

    d1 = _riemann_distance_batch(
        X,
        M1_inv_sqrt,
    )

    distances = np.column_stack([d0, d1])

    scores = -distances ** 2

    scores -= scores.max(
        axis=1,
        keepdims=True,
    )

    probs = np.exp(scores)

    probs /= probs.sum(
        axis=1,
        keepdims=True,
    )

    return probs[:, target_class_idx]

def _precompute_masks(n_channels):
    n_subsets = 2 ** n_channels
    idx = np.arange(n_subsets, dtype=np.int64)

    bits = (
        (idx[:, None] >> np.arange(n_channels, dtype=np.int64)) & 1
    ).astype(bool)

    return idx, bits


def _precompute_size_weights(n_channels):
    n_fact = math.factorial(n_channels)

    return np.array(
        [
            math.factorial(s)
            * math.factorial(n_channels - s - 1)
            / n_fact
            for s in range(n_channels)
        ],
        dtype=np.float64,
    )


def _score_all_subsets(
    x,
    bits,
    clf,
    target_class_idx,
    batch_size,
    baseline=None,
):
    """
    Compute v(S) for all subsets S.

    Returns
    -------
    v : ndarray, shape (2**n,)
        v[S] = classifier probability for coalition S.
    """
    print("New sample")

    n = x.shape[0]
    n_subsets = bits.shape[0]

    v = np.empty(n_subsets, dtype=np.float64)

    diag = np.arange(n)

    for start in range(0, n_subsets, batch_size):
        stop = min(start + batch_size, n_subsets)

        m = bits[start:stop]  # (batch, n)

        # Keep an element (a,b) only if both channels
        # are present in the coalition.
        keep = m[:, :, None] & m[:, None, :]

        # (batch, n, n)
        coalitions = x[None, :, :] * keep

        # Put baseline on the diagonal for excluded channels.
        if baseline is not None:
            not_in_s = ~m

            coalitions[:, diag, diag] += (
                not_in_s * baseline[None, :]
            )

        # Project the whole batch onto SPD.
        # proj_on_spd must support shape (batch, n, n).
        t0 = time.perf_counter()
        coalitions = proj_on_spd(coalitions)
        t1 = time.perf_counter()

        # Safety check while debugging.
        assert coalitions.shape == (
            stop - start,
            n,
            n,
        ), (
            f"Wrong shape after proj_on_spd: "
            f"{coalitions.shape}, "
            f"expected {(stop - start, n, n)}"
        )

        covmeans = clf.covmeans_

        probs = mdm_predict_proba_fast(
            coalitions,
            covmeans,
            target_class_idx=0,
        )
        t2 = time.perf_counter()
        print(f"proj_on_spd: {t1-t0:.3f}s, predict: {t2-t1:.3f}s")
        v[start:stop] = probs


    return v


def _compute_phi_from_v(
    v,
    n,
    weight_by_size,
    popcount,
):
    """
    Compute exact Shapley values from v(S).
    """

    phi = np.zeros(n, dtype=np.float64)

    idx = np.arange(len(v), dtype=np.int64)

    for i in range(n):
        bit = 1 << i

        # Coalitions S that do not contain channel i.
        no_i = idx[(idx & bit) == 0]

        # S union {i}.
        with_i = no_i | bit

        # |S|
        sizes = popcount[no_i]

        weights = weight_by_size[sizes]

        marginal = v[with_i] - v[no_i]

        phi[i] = np.dot(weights, marginal)

    return phi


def shapley_values_covariances_parallel(
    C_train,
    C_test,
    clf,
    baseline,
    target_class_idx=0,
    batch_size=100_000,
    n_jobs=-1,
):
    """
    Exact Shapley values for covariance matrices.

    Parallelization is performed across test samples.
    """

    n = C_test.shape[1]

    baseline = np.asarray(baseline, dtype=np.float64)

    if baseline.shape != (n,):
        raise ValueError(
            f"baseline must have shape ({n},), "
            f"got {baseline.shape}"
        )

    # Subset masks
    _, bits = _precompute_masks(n)

    # Number of channels in each subset
    popcount = bits.sum(axis=1)

    # Shapley weights by coalition size
    weight_by_size = _precompute_size_weights(n)

    def process_one(x):
        v = _score_all_subsets(
            x,
            bits,
            clf,
            target_class_idx,
            batch_size,
            baseline=baseline,
        )

        return _compute_phi_from_v(
            v,
            n,
            weight_by_size,
            popcount,
        )

    results = Parallel(
        n_jobs=n_jobs,
        prefer="processes",
    )(
        delayed(process_one)(x)
        for x in C_test
    )

    return np.asarray(results)