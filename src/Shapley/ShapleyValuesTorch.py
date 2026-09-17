import torch
import numpy as np
from joblib import Parallel, delayed
import math
import time


def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def inv_sqrt_spd_torch(M, device, dtype=torch.float32):
    """
    M^{-1/2} for a single SPD matrix M (numpy array, shape (n, n)).
    Computed ONCE per run (covmeans don't change across samples/batches).
    """
    M_t = torch.as_tensor(M, dtype=dtype, device=device)
    eigvals, eigvecs = torch.linalg.eigh(M_t)
    eigvals = torch.clamp(eigvals, min=1e-12)
    return (eigvecs * (1.0 / torch.sqrt(eigvals))[None, :]) @ eigvecs.T


def riemann_distance_batch_torch(X, M_inv_sqrt):
    """
    X          : (batch, n, n) torch tensor, already on the target device
    M_inv_sqrt : (n, n) torch tensor, already on the target device

    Returns
    -------
    distances : (batch,) torch tensor
    """
    whitened = M_inv_sqrt[None, :, :] @ X @ M_inv_sqrt[None, :, :]

    # Batched eigendecomposition -- this is the part that parallelizes
    # across the whole batch on GPU instead of looping matrix-by-matrix.
    eigvals = torch.linalg.eigvalsh(whitened)
    eigvals = torch.clamp(eigvals, min=1e-12)

    distances = torch.sqrt(torch.sum(torch.log(eigvals) ** 2, dim=1))
    return distances


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



def mdm_predict_proba_fast_torch(X, M0_inv_sqrt, M1_inv_sqrt, target_class_idx=0):
    """
    X: (batch, n, n) torch tensor on the target device.
    M0_inv_sqrt / M1_inv_sqrt: precomputed only once outside the batch loop.

    Returns a numpy array of shape (batch,) -- moved back to CPU here
    since downstream Shapley bookkeeping (v[start:stop] = ...) is numpy.
    """
    d0 = riemann_distance_batch_torch(X, M0_inv_sqrt)
    d1 = riemann_distance_batch_torch(X, M1_inv_sqrt)

    distances = torch.stack([d0, d1], dim=1)
    scores = -distances ** 2
    scores = scores - scores.max(dim=1, keepdim=True).values
    probs = torch.exp(scores)
    probs = probs / probs.sum(dim=1, keepdim=True)

    return probs[:, target_class_idx].detach().cpu().numpy()


def coalitions_to_gpu(coalitions_np, device, dtype=torch.float32):
    """Move a (batch, n, n) numpy array of coalition matrices to the GPU."""
    return torch.as_tensor(coalitions_np, dtype=dtype, device=device)


def score_batch_gpu(coalitions_np, M0_inv_sqrt, M1_inv_sqrt, device,
                     dtype=torch.float32, target_class_idx=0, epsilon=1e-6):
    """
    Full replacement for the per-batch body inside _score_all_subsets:
    move to GPU -> compute MDM probabilities -> back to numpy.

    coalitions_np : (batch, n, n) numpy array (already masked/baseline-filled)
    Returns a numpy array of shape (batch,).
    """
    X = coalitions_to_gpu(coalitions_np, device, dtype=dtype)
    #t0 = time.perf_counter()
    #X = proj_on_spd_torch(X, epsilon=epsilon)
    #t1 = time.perf_counter()
    probs = mdm_predict_proba_fast_torch(
        X, M0_inv_sqrt, M1_inv_sqrt, target_class_idx=target_class_idx
    )
    #t2 = time.perf_counter()
    #print(f"proj_on_spd: {t1-t0:.3f}s, predict: {t2-t1:.3f}s")
    return probs

device = get_device()  
print(device)

def _score_all_subsets(x, bits, clf, target_class_idx, batch_size, baseline=None,
                        M0_inv_sqrt=None, M1_inv_sqrt=None, device=None):
    n = x.shape[0]
    n_subsets = bits.shape[0]
    v = np.empty(n_subsets, dtype=np.float64)
    diag = np.arange(n)

    for start in range(0, n_subsets, batch_size):
        stop = min(start + batch_size, n_subsets)
        m = bits[start:stop]
        keep = m[:, :, None] & m[:, None, :]
        coalitions = x[None, :, :] * keep
        if baseline is not None:
            not_in_s = ~m
            coalitions[:, diag, diag] += not_in_s * baseline[None, :]

        probs = score_batch_gpu(coalitions, M0_inv_sqrt, M1_inv_sqrt, device)
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
    batch_size=20_000,
    n_jobs=1,
):
    """
    Exact Shapley values for covariance matrices.

    Parallelization is performed across test samples.

    Baseline is a numpy array of shape (n_channels,) representing the baseline covariance 
    matrix (diagonal) to use for missing channels. 
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

    covmeans = clf.covmeans_
    i = 0

    def process_one(x):
        nonlocal i
        i +=1
        print("Processing sample ",i)
        v = _score_all_subsets(
            x,
            bits,
            clf,
            target_class_idx,
            batch_size,
            M0_inv_sqrt=inv_sqrt_spd_torch(covmeans[0], device), 
            M1_inv_sqrt=inv_sqrt_spd_torch(covmeans[1], device),
            device = device, 
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