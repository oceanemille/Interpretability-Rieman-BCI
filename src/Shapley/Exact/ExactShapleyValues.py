"""Fast exact Shapley values for covariance-matrix classifiers.

Exact computation necessarily scores ``2**n`` coalitions for ``n`` channels.
This module keeps coalition construction, scoring, and Shapley reduction on a
single Torch device. Only input covariances and final results cross devices.
Fitted Torch models such as SPDNet are evaluated directly in large batches.

MDM and TangentSpace Classifier logic are taken from pyriemann but the
models are reimplemented in order to be boosted by 
Torch and optimisations to limit computational time
"""

import math
import copy

import numpy as np
import torch


DEFAULT_LINALG_BATCH_SIZE = 20_000
MIN_EIGENVALUE = 1e-12


def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


@torch.inference_mode()
def inv_sqrt_spd_torch(
    matrix,
    device,
    dtype = torch.float32):
    """Return the inverse square root of one symmetric positive matrix."""
    matrix_t = torch.as_tensor(matrix, dtype=dtype, device=device)
    eigvals, eigvecs = torch.linalg.eigh(matrix_t)
    eigvals.clamp_(min=MIN_EIGENVALUE)
    return (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T


def _squared_riemann_distance(
    matrices,
    reference_inv_sqrt):
    """Squared affine-invariant distance to one reference matrix."""
    reference = reference_inv_sqrt.unsqueeze(0)
    whitened = reference @ matrices @ reference
    eigvals = torch.linalg.eigvalsh(whitened)
    eigvals.clamp_(min=MIN_EIGENVALUE)
    return torch.log(eigvals).square().sum(dim=-1)


@torch.inference_mode()
def _score_mdm(
    coalitions,
    references_inv_sqrt,
    target_class_idx,
    linalg_batch_size):
    """Score a coalition batch with an MDM-like classifier."""
    n_classes = len(references_inv_sqrt)
    if not 0 <= target_class_idx < n_classes:
        raise ValueError(
            f"target_class_idx must be in [0, {n_classes}), got {target_class_idx}"
        )

    output = torch.empty(
        coalitions.shape[0], dtype=coalitions.dtype, device=coalitions.device
    )
    for start in range(0, coalitions.shape[0], linalg_batch_size):
        stop = min(start + linalg_batch_size, coalitions.shape[0])
        chunk = coalitions[start:stop]

        if n_classes == 2:
            d0_sq = _squared_riemann_distance(chunk, references_inv_sqrt[0])
            d1_sq = _squared_riemann_distance(chunk, references_inv_sqrt[1])
            p0 = torch.sigmoid(d1_sq - d0_sq)
            output[start:stop] = p0 if target_class_idx == 0 else 1.0 - p0
            continue

        scores = torch.empty(
            (stop - start, n_classes), dtype=coalitions.dtype, device=coalitions.device
        )
        for class_idx, reference in enumerate(references_inv_sqrt):
            scores[:, class_idx] = -_squared_riemann_distance(chunk, reference)
        output[start:stop] = torch.softmax(scores, dim=1)[:, target_class_idx]

    return output


def _extract_tangentspace_and_classifier(clf):
    """Extract a fitted TangentSpace transform and its final classifier."""
    pipe = getattr(clf, "_pipe", clf)
    steps = list(pipe.named_steps.values()) if hasattr(pipe, "named_steps") else list(pipe)
    tangent_step = next((step for step in steps if hasattr(step, "reference_")), None)
    if tangent_step is None:
        raise ValueError(
            "Could not find a fitted TangentSpace step exposing `reference_`."
        )

    classifier = steps[-1]
    if not hasattr(classifier, "coef_") or not hasattr(classifier, "intercept_"):
        raise TypeError(
            "The TangentSpace classifier must be a fitted linear classifier "
            "exposing `coef_` and `intercept_`."
        )
    return tangent_step.reference_, classifier


def _tangent_classifier_matrices(
    classifier,
    n_channels,
    device,
    dtype):
    """Convert tangent-vector coefficients to symmetric matrix coefficients.

    This lets the linear classifier consume eigenpairs directly, avoiding the
    construction and vectorization of every matrix logarithm.
    """
    coef = torch.as_tensor(classifier.coef_, dtype=dtype, device=device)
    intercept = torch.as_tensor(classifier.intercept_, dtype=dtype, device=device)
    expected_features = n_channels * (n_channels + 1) // 2
    if coef.ndim != 2 or coef.shape[1] != expected_features:
        raise ValueError(
            "Classifier coefficient width does not match the TangentSpace "
            f"dimension: expected {expected_features}, got {coef.shape[-1]}."
        )

    row, col = torch.triu_indices(n_channels, n_channels, device=device)
    weights = torch.zeros(
        (coef.shape[0], n_channels, n_channels), dtype=dtype, device=device
    )
    values = coef / math.sqrt(2.0)
    diagonal = row == col
    values[:, diagonal] = coef[:, diagonal]
    weights[:, row, col] = values
    weights[:, col, row] = values
    return weights, intercept


@torch.inference_mode()
def _score_tangent_linear(
    coalitions,
    reference_inv_sqrt,
    classifier_matrices,
    intercept,
    target_class_idx,
    linalg_batch_size):
    """Score TangentSpace coalitions without materializing their logarithms."""
    n_outputs = classifier_matrices.shape[0]
    n_classes = 2 if n_outputs == 1 else n_outputs
    if not 0 <= target_class_idx < n_classes:
        raise ValueError(
            f"target_class_idx must be in [0, {n_classes}), got {target_class_idx}"
        )

    output = torch.empty(
        coalitions.shape[0], dtype=coalitions.dtype, device=coalitions.device
    )
    reference = reference_inv_sqrt.unsqueeze(0)

    for start in range(0, coalitions.shape[0], linalg_batch_size):
        stop = min(start + linalg_batch_size, coalitions.shape[0])
        whitened = reference @ coalitions[start:stop] @ reference
        eigvals, eigvecs = torch.linalg.eigh(whitened)
        eigvals.clamp_(min=MIN_EIGENVALUE)

        # diag(Q.T @ W @ Q), for each sample and classifier output.
        projected_diagonal = torch.einsum(
            "bik,cij,bjk->bck", eigvecs, classifier_matrices, eigvecs
        )
        logits = (projected_diagonal * torch.log(eigvals).unsqueeze(1)).sum(-1)
        logits.add_(intercept)

        if n_outputs == 1:
            p1 = torch.sigmoid(logits[:, 0])
            output[start:stop] = p1 if target_class_idx == 1 else 1.0 - p1
        else:
            output[start:stop] = torch.softmax(logits, dim=1)[:, target_class_idx]

    return output


@torch.inference_mode()
def _score_torch_model(
    coalitions,
    model,
    target_class_idx,
    inference_batch_size,
):
    """Score SPD coalitions with a Torch classifier returning class logits.

    SPDNet and similar deep Riemannian networks already perform batched matrix
    operations. Feeding them contiguous chunks avoids both the Python loop over
    coalitions and any CPU/device transfers in the exponential part of the
    exact computation.
    """
    output = torch.empty(
        coalitions.shape[0], dtype=coalitions.dtype, device=coalitions.device
    )

    for start in range(0, coalitions.shape[0], inference_batch_size):
        stop = min(start + inference_batch_size, coalitions.shape[0])
        logits = model(coalitions[start:stop])
        if not isinstance(logits, torch.Tensor):
            raise TypeError("The Torch classifier must return a tensor of logits.")

        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            if target_class_idx not in (0, 1):
                raise ValueError(
                    "target_class_idx must be 0 or 1 for a binary one-logit model"
                )
            positive = torch.sigmoid(logits.reshape(-1))
            output[start:stop] = (
                positive if target_class_idx == 1 else 1.0 - positive
            )
            continue

        if logits.ndim != 2:
            raise ValueError(
                "The Torch classifier must return logits with shape "
                f"(batch, classes), got {tuple(logits.shape)}."
            )
        if not 0 <= target_class_idx < logits.shape[1]:
            raise ValueError(
                "target_class_idx must be in "
                f"[0, {logits.shape[1]}), got {target_class_idx}"
            )

        # For two classes this is exactly softmax(logits)[:, target], with one
        # exponential instead of two. Keep softmax for the multiclass case.
        if logits.shape[1] == 2:
            other_class_idx = 1 - target_class_idx
            output[start:stop] = torch.sigmoid(
                logits[:, target_class_idx] - logits[:, other_class_idx]
            )
        else:
            output[start:stop] = torch.softmax(logits, dim=1)[:, target_class_idx]

    return output


def _make_scorer(
    clf,
    device,
    dtype,
    linalg_batch_size,
):
    """Prepare classifier constants once and return a device scorer."""
    if isinstance(clf, torch.nn.Module):
        # SPDNet has no sklearn-style fitted attributes. Move it once, before
        # enumerating coalitions, and use eval mode so batch normalization and
        # other stateful layers are deterministic for every chunk size.
        model = clf.to(device=device, dtype=dtype)
        model.eval()

        def score(coalitions, target_class_idx):
            return _score_torch_model(
                coalitions,
                model,
                target_class_idx,
                linalg_batch_size,
            )

        return score

    if hasattr(clf, "covmeans_"):
        references = tuple(
            inv_sqrt_spd_torch(matrix, device, dtype) for matrix in clf.covmeans_
        )

        def score(coalitions, target_class_idx):
            return _score_mdm(
                coalitions, references, target_class_idx, linalg_batch_size
            )

        return score

    reference, classifier = _extract_tangentspace_and_classifier(clf)
    reference_inv_sqrt = inv_sqrt_spd_torch(reference, device, dtype)
    classifier_matrices, intercept = _tangent_classifier_matrices(
        classifier, reference_inv_sqrt.shape[0], device, dtype
    )

    def score(coalitions, target_class_idx):
        return _score_tangent_linear(
            coalitions,
            reference_inv_sqrt,
            classifier_matrices,
            intercept,
            target_class_idx,
            linalg_batch_size,
        )

    return score


@torch.inference_mode()
def _precompute_masks(
    n_channels,
    device,
    construction_batch_size=1_000_000,
):
    """Create all exact coalition masks and cardinalities on-device."""
    if n_channels >= 63:
        raise ValueError(
            "Exact enumeration with 63 or more channels cannot use Torch's "
            "signed 64-bit indices and is computationally infeasible."
        )

    n_subsets = 1 << n_channels
    masks = torch.empty((n_subsets, n_channels), dtype=torch.bool, device=device)
    shifts = torch.arange(n_channels, dtype=torch.int64, device=device)
    for start in range(0, n_subsets, construction_batch_size):
        stop = min(start + construction_batch_size, n_subsets)
        indices = torch.arange(start, stop, dtype=torch.int64, device=device)
        masks[start:stop] = ((indices.unsqueeze(1) >> shifts) & 1).bool()

    popcount = masks.sum(dim=1).to(torch.int16)
    return masks, popcount


def _size_weights(
    n_channels,
    device,
) :
    """Return s!(n-s-1)!/n! for every coalition size s."""
    weights = [
        1.0 / (n_channels * math.comb(n_channels - 1, size))
        for size in range(n_channels)
    ]
    return torch.tensor(weights, dtype=torch.float64, device=device)


@torch.inference_mode()
def _score_all_subsets(
    covariance,
    masks,
    baseline,
    score,
    target_class_idx,
    batch_size,
    dtype,
):
    """Construct and score every coalition while keeping data on the device."""
    covariance_t = torch.as_tensor(covariance, dtype=dtype, device=masks.device)
    n_subsets = masks.shape[0]
    values = torch.empty(n_subsets, dtype=torch.float64, device=masks.device)

    for start in range(0, n_subsets, batch_size):
        stop = min(start + batch_size, n_subsets)
        present = masks[start:stop]

        # Zero absent rows/columns without storing all full coalition masks.
        coalitions = covariance_t.unsqueeze(0) * present.unsqueeze(2)
        coalitions.mul_(present.unsqueeze(1))
        coalitions.diagonal(dim1=-2, dim2=-1).add_(~present * baseline)
        values[start:stop] = score(coalitions, target_class_idx)

    return values


@torch.inference_mode()
def _compute_shapley_values(
    values,
    popcount,
    size_weights,
    n_channels,
):
    """Reduce all v(S) scores to exact Shapley values on the device."""
    result = torch.empty(n_channels, dtype=torch.float64, device=values.device)
    for channel in range(n_channels):
        half_block = 1 << channel
        value_blocks = values.reshape(-1, 2, half_block)
        count_blocks = popcount.reshape(-1, 2, half_block)
        marginal = value_blocks[:, 1] - value_blocks[:, 0]
        weights = size_weights[count_blocks[:, 0].long()]
        result[channel] = torch.sum(weights * marginal)
    return result


@torch.inference_mode()
def shapley_values_covariances_parallel(
    C_train,
    C_test,
    clf,
    baseline,
    target_class_idx = 0,
    batch_size = 100_000,
    n_jobs = 1,
    *,
    compute_device=None,
    dtype = torch.float32,
    linalg_batch_size = DEFAULT_LINALG_BATCH_SIZE,
    verbose = True,
):
    """Compute exact channel Shapley values for covariance matrices.

    ``clf`` can be a fitted pyRiemann MDM, a fitted TangentSpace pipeline ending
    in a linear classifier, or a fitted Torch module such as SPDNet that accepts
    a batch of SPD matrices and returns class logits. ``C_train`` remains in the
    signature for compatibility, but is unnecessary after fitting.

    ``batch_size`` controls coalition construction memory. For Torch models,
    ``linalg_batch_size`` is also the maximum forward-pass batch size. The model
    is moved to ``compute_device`` with ``dtype`` and left there in eval mode.
    """
    del C_train
    if n_jobs != 1:
        raise ValueError("n_jobs must be 1; exact batches already saturate one device")
    if batch_size <= 0 or linalg_batch_size <= 0:
        raise ValueError("batch_size and linalg_batch_size must be positive")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")

    selected_device = torch.device(compute_device) if compute_device else get_device()
    if isinstance(C_test, torch.Tensor):
        C_test = C_test.detach().cpu().numpy()
    else:
        C_test = np.asarray(C_test)
    if C_test.ndim != 3 or C_test.shape[1] != C_test.shape[2]:
        raise ValueError(
            f"C_test must have shape (samples, channels, channels), got {C_test.shape}"
        )
    n_channels = C_test.shape[1]
    if isinstance(baseline, torch.Tensor):
        baseline_np = baseline.detach().cpu().numpy()
    else:
        baseline_np = np.asarray(baseline)
    if baseline_np.shape != (n_channels,):
        raise ValueError(
            f"baseline must have shape ({n_channels},), got {baseline_np.shape}"
        )

    baseline_t = torch.tensor(baseline_np, dtype=dtype, device=selected_device)
    masks, popcount = _precompute_masks(n_channels, selected_device)
    weights = _size_weights(n_channels, selected_device)
    score = _make_scorer(clf, selected_device, dtype, linalg_batch_size)

    results = np.empty((len(C_test), n_channels), dtype=np.float64)
    for sample_idx, covariance in enumerate(C_test):
        if verbose:
            print(f"Processing sample {sample_idx + 1}/{len(C_test)}")
        values = _score_all_subsets(
            covariance,
            masks,
            baseline_t,
            score,
            target_class_idx,
            batch_size,
            dtype,
        )
        shapley = _compute_shapley_values(values, popcount, weights, n_channels)
        results[sample_idx] = shapley.cpu().numpy()

    return results


def _resolve_classifier(clf):
    """Resolve the public classifier shortcuts without importing them eagerly."""
    from pyriemann.classification import MDM, TSClassifier
    from src.SPDNet.SPDNet import SPDNetBatchNorm

    if clf is None:
        return SPDNetBatchNorm, True
    if isinstance(clf, str):
        shortcuts = {
            "SPDNet": (SPDNetBatchNorm, True),
            "MDM": (MDM, False),
            "TSClassifier": (TSClassifier, False),
        }
        if clf not in shortcuts:
            raise ValueError(
                f"Unknown classifier {clf!r}; expected one of {tuple(shortcuts)}."
            )
        return shortcuts[clf]
    return clf, clf is SPDNetBatchNorm


def _is_torch_classifier(clf):
    """Return whether a classifier instance or class is a Torch module."""
    if isinstance(clf, torch.nn.Module):
        return True
    return isinstance(clf, type) and issubclass(clf, torch.nn.Module)


def _fresh_classifier(
    clf,
    *,
    is_default_spdnet,
    n_channels,
    n_classes,
    model_config,
):
    """Create an unfitted classifier for one reproducible split."""
    if isinstance(clf, torch.nn.Module):
        if model_config:
            raise ValueError(
                "model_config cannot be used with an already instantiated "
                "Torch model; pass its class or a factory instead."
            )
        return copy.deepcopy(clf)

    if _is_torch_classifier(clf):
        config = dict(model_config or {})
        if is_default_spdnet:
            defaults = {
                "input_type": "cov",
                "n_chans": n_channels,
                "subspacedim": min(16, n_channels),
                "n_outputs": n_classes,
                "bn": None,
            }
            defaults.update(config)
            config = defaults
        return clf(**config)

    if isinstance(clf, type):
        return clf(**(model_config or {}))

    try:
        from sklearn.base import clone

        return clone(clf)
    except Exception:
        if callable(clf):
            return clf(**(model_config or {}))
        return copy.deepcopy(clf)


def _covariance_baseline(covariances, labels):
    """Reproduce the historical diagonal, class-balanced Riemannian baseline."""
    from pyriemann.utils.mean import mean_riemann

    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("At least two classes are required to compute the baseline.")
    class_means = [
        mean_riemann(covariances[labels == label]) for label in classes
    ]
    return np.diag(mean_riemann(np.asarray(class_means)))


def _torch_predictions(logits):
    """Convert one-logit or multiclass outputs to encoded class predictions."""
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        return (logits.reshape(-1) >= 0).long()
    if logits.ndim != 2:
        raise ValueError(
            "A Torch classifier must return one logit per sample or a "
            f"(samples, classes) tensor, got {tuple(logits.shape)}."
        )
    return logits.argmax(dim=1)


def _train_torch_classifier(
    model,
    C_train,
    y_train,
    C_test,
    y_test,
    *,
    epochs,
    lr,
    verbose,
):
    """Train a deep covariance classifier with the historical training recipe."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(C_train)
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError(
                "Training requires a Torch model returning at least two class "
                "logits. A pretrained one-logit model can still be explained "
                "with pretrained=True."
            )
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.inference_mode():
                accuracy = (
                    _torch_predictions(model(C_test)) == y_test
                ).float().mean().item()
            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f} "
                f"- Test Acc: {accuracy:.4f}"
            )

    model.eval()
    return model


def _encode_labels(y_train, y_test):
    """Fit the historical alphabetical label mapping on the training fold."""
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    train_encoded = encoder.fit_transform(y_train)
    test_encoded = encoder.transform(y_test)
    return train_encoded, test_encoded, encoder.classes_


def _as_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _looks_like_covariances(X):
    return X.ndim == 3 and X.shape[1] == X.shape[2]


def _transform_covariances(X_train, X_test, input_is_covariance):
    """Transform raw EEG into covariances, or retain covariance input."""
    if input_is_covariance:
        return np.asarray(X_train), np.asarray(X_test)

    from pyriemann.estimation import Covariances

    transformer = Covariances()
    C_train = transformer.fit_transform(X_train)
    if len(X_test) == 0:
        n_channels = C_train.shape[1]
        return C_train, np.empty((0, n_channels, n_channels), dtype=C_train.dtype)
    return C_train, transformer.transform(X_test)


def _classifier_accuracy(clf, C_test, y_test, compute_device, dtype):
    """Evaluate either a Torch model or a fitted sklearn-style classifier."""
    if isinstance(clf, torch.nn.Module):
        model = clf.to(device=compute_device, dtype=dtype)
        model.eval()
        C_test_t = torch.as_tensor(C_test, dtype=dtype, device=compute_device)
        y_test_t = torch.as_tensor(y_test, dtype=torch.long, device=compute_device)
        with torch.inference_mode():
            return (
                _torch_predictions(model(C_test_t)) == y_test_t
            ).float().mean().item()
    return float(clf.score(C_test, y_test))


def fit(
    X,
    y,
    clf=None,
    n_splits=10,
    *,
    pretrained=False,
    input_is_covariance=None,
    baseline=None,
    model_config=None,
    test_size=0.2,
    epochs=200,
    lr=1e-3,
    target_class_idx=0,
    compute_device=None,
    dtype=torch.float32,
    batch_size=20_000,
    linalg_batch_size=DEFAULT_LINALG_BATCH_SIZE,
    verbose=True,
):
    """Fit a classifier and compute exact channel Shapley values.

    ``clf`` accepts ``"SPDNet"``, ``"MDM"``, ``"TSClassifier"``, a classifier
    class/factory, or an estimator/module instance. ``model_config`` supplies
    constructor arguments for a class or factory. Torch models are trained with
    full-batch Adam and cross-entropy; sklearn-style classifiers use ``fit``.

    Set ``pretrained=True`` to explain an already fitted classifier instance.
    In that mode no split, cloning, or training occurs: ``X`` and ``y`` are the
    evaluation set. Supply the training-derived diagonal ``baseline`` whenever
    possible. If it is omitted, a baseline is estimated from the evaluation
    data for convenience.

    ``input_is_covariance`` defaults to automatic detection: square trials are
    treated as covariance matrices, while non-square trials are transformed by
    pyRiemann's ``Covariances`` estimator.

    Returns
    -------
    shapley_values : list of ndarray
        One ``(n_test_trials, n_channels)`` array per split, or a one-element
        list in pretrained mode.
    scores : ndarray
        Classification accuracy for every corresponding result.
    """
    X_np = _as_numpy(X)
    y_np = _as_numpy(y)
    if X_np.ndim != 3:
        raise ValueError(
            "X must have shape (trials, channels, times) or "
            f"(trials, channels, channels), got {X_np.shape}."
        )
    if y_np.ndim != 1 or len(y_np) != len(X_np):
        raise ValueError(
            f"y must have shape ({len(X_np)},), got {y_np.shape}."
        )
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")

    if input_is_covariance is None:
        input_is_covariance = _looks_like_covariances(X_np)
    selected_device = (
        torch.device(compute_device) if compute_device is not None else get_device()
    )
    classifier_spec, is_default_spdnet = _resolve_classifier(clf)

    if pretrained:
        if isinstance(classifier_spec, type) or not (
            isinstance(classifier_spec, torch.nn.Module)
            or hasattr(classifier_spec, "predict")
        ):
            raise TypeError(
                "pretrained=True requires an already fitted classifier instance."
            )

        C_eval, _ = _transform_covariances(
            X_np, X_np[:0], input_is_covariance
        )
        if isinstance(classifier_spec, torch.nn.Module):
            _, y_eval, classes = _encode_labels(y_np, y_np)
        else:
            y_eval = y_np
            classes = getattr(classifier_spec, "classes_", np.unique(y_np))
        reference = (
            _as_numpy(baseline)
            if baseline is not None
            else _covariance_baseline(C_eval, y_np)
        )
        score = _classifier_accuracy(
            classifier_spec, C_eval, y_eval, selected_device, dtype
        )
        values = shapley_values_covariances_parallel(
            None,
            C_eval,
            classifier_spec,
            reference,
            target_class_idx=target_class_idx,
            batch_size=batch_size,
            compute_device=selected_device,
            dtype=dtype,
            linalg_batch_size=linalg_batch_size,
            verbose=verbose,
        )
        return [values], np.asarray([score], dtype=float)

    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1")

    from sklearn.model_selection import train_test_split

    all_shapley_values = []
    scores = []
    for i in range(n_splits):
        if verbose:
            print(f"Split {i + 1}/{n_splits} (split={i})")
        X_train, X_test, y_train, y_test = train_test_split(
            X_np,
            y_np,
            test_size=test_size,
            stratify=y_np,
            random_state=i,
        )
        C_train, C_test = _transform_covariances(
            X_train, X_test, input_is_covariance
        )
        reference = (
            _as_numpy(baseline)
            if baseline is not None
            else _covariance_baseline(C_train, y_train)
        )

        torch.manual_seed(i)
        if selected_device.type == "cuda":
            torch.cuda.manual_seed_all(i)

        y_train_encoded, y_test_encoded, classes = _encode_labels(y_train, y_test)
        fitted_clf = _fresh_classifier(
            classifier_spec,
            is_default_spdnet=is_default_spdnet,
            n_channels=C_train.shape[1],
            n_classes=len(classes),
            model_config=model_config,
        )

        if isinstance(fitted_clf, torch.nn.Module):
            fitted_clf = fitted_clf.to(device=selected_device, dtype=dtype)
            C_train_fitted = torch.as_tensor(
                C_train, dtype=dtype, device=selected_device
            )
            C_test_fitted = torch.as_tensor(
                C_test, dtype=dtype, device=selected_device
            )
            y_train_fitted = torch.as_tensor(
                y_train_encoded, dtype=torch.long, device=selected_device
            )
            y_test_fitted = torch.as_tensor(
                y_test_encoded, dtype=torch.long, device=selected_device
            )
            _train_torch_classifier(
                fitted_clf,
                C_train_fitted,
                y_train_fitted,
                C_test_fitted,
                y_test_fitted,
                epochs=epochs,
                lr=lr,
                verbose=verbose,
            )
            score = _classifier_accuracy(
                fitted_clf,
                C_test,
                y_test_encoded,
                selected_device,
                dtype,
            )
        else:
            fitted_clf.fit(C_train, y_train)
            score = _classifier_accuracy(
                fitted_clf, C_test, y_test, selected_device, dtype
            )
                
        values = shapley_values_covariances_parallel(
            C_train,
            C_test,
            fitted_clf,
            reference,
            target_class_idx=target_class_idx,
            batch_size=batch_size,
            compute_device=selected_device,
            dtype=dtype,
            linalg_batch_size=linalg_batch_size,
            verbose=verbose,
        )

        all_shapley_values.append(values)
        scores.append(score)

    return all_shapley_values, np.asarray(scores, dtype=float)