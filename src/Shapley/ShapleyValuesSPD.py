import warnings
import numpy as np
import torch
from functools import partial
from shap import KernelExplainer
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

from src.SPDNet.SPDNet import SPDNetBatchNorm, train_model
from .ShapleyValuesEEG import ShapRecorder


def _as_covariances(X):
    """Return covariance matrices and reject raw EEG input explicitly."""
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    if X.ndim != 3 or X.shape[1] != X.shape[2]:
        raise ValueError(
            "X must contain covariance matrices with shape "
            f"(trials, channels, channels), got {X.shape}."
        )
    return X


def _compute_cov_baseline(C_train, y_train):
    """Return the diagonal class-balanced Riemannian reference matrix."""
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError("At least two classes are required for the baseline.")
    class_means = [mean_riemann(C_train[y_train == c]) for c in classes]
    balanced_mean = mean_riemann(np.asarray(class_means))
    return np.diag(np.diag(balanced_mean))


def reconstruct_shap_samples(mask_2d, current_run_cov, reference_cov):
    """Reconstruct covariance matrices from SHAP coalition masks."""
    n_simulations, n_channels = mask_2d.shape
    inactive = mask_2d <= 0.5 

    C_reconstructed = np.tile(current_run_cov, (n_simulations, 1, 1)).astype(current_run_cov.dtype)

    row_mask = inactive[:, :, None]   
    col_mask = inactive[:, None, :]   
    reference = reference_cov[None, :, :]

    C_reconstructed = np.where(row_mask, reference, C_reconstructed)
    C_reconstructed = np.where(col_mask, reference, C_reconstructed)

    return C_reconstructed


class ShapleyValuesSPD:
    """KernelSHAP channel attribution computed on covariance matrices."""

    def fit(self, X, y, n_splits=10, model=None, model_trained=None,
            baseline=None, n_samples=2000, X_train=None, y_train=None):
        """Compute Shapley values for a classifier on covariance matrices."""
        X = _as_covariances(X)
        y = np.asarray(y)

        if model_trained is not None:
            if n_splits != 10:
                warnings.warn("`n_splits` is ignored with a pre-trained model.")

            if baseline is None:
                reference_X = X if X_train is None else _as_covariances(X_train)
                reference_y = y if y_train is None else np.asarray(y_train)
                split_baseline = _compute_cov_baseline(reference_X, reference_y)
            else:
                split_baseline = baseline
            shap_values, ratios = self.KernelShap(
                X, X, model_trained, split_baseline, n_samples
            )
            score = np.mean(model_trained.predict(X) == y)

            self.shap_values_ = [shap_values]
            self.ratios_ = [ratios]
            self.scores_ = np.array([score])
        else:
            if model is None:
                raise ValueError("Either give `model_trained` (already trained), or `model` (will be trained by this method).")

            all_shap_values, all_ratios, all_scores = [], [], []
            for i in range(n_splits):
                print("Split",i)
                C_train, C_test, train_y, test_y = train_test_split(
                    X, y, train_size=0.8, stratify=y, random_state=i
                )

                clf = clone(model)
                clf.fit(C_train, train_y)

                split_baseline = (
                    baseline if baseline is not None
                    else _compute_cov_baseline(C_train, train_y)
                )
                shap_values, ratios = self.KernelShap(C_train, C_test, clf, split_baseline, n_samples)

                all_shap_values.append(shap_values)
                all_ratios.append(ratios)
                all_scores.append(clf.score(C_test, test_y))

            self.shap_values_ = all_shap_values
            self.ratios_ = all_ratios
            self.scores_ = np.array(all_scores)

        return self

    def _stable_predict(self, mask_2d, current_run_cov, reference_cov, pipeline, recorder=None):
        C_reconstructed = reconstruct_shap_samples(mask_2d, current_run_cov, reference_cov)
        predictions = pipeline.predict_proba(C_reconstructed)[:, 0]

        if recorder is not None and hasattr(pipeline, "covmeans_"):
            C0, C1 = pipeline.covmeans_[0], pipeline.covmeans_[1]
            d0 = np.array([distance_riemann(c, C0) for c in C_reconstructed])
            d1 = np.array([distance_riemann(c, C1) for c in C_reconstructed])
            dc = distance_riemann(C0, C1)
            ratios = (d0 - d1) / dc
            recorder.add(mask_2d, ratios, predictions)

        return predictions

    def _compute_cov_baseline(self, C_train, y_train):
        """Baseline = diagonal matrix built from the Fréchet mean for each class (for n_classes)"""
        return _compute_cov_baseline(C_train, y_train)

    def KernelShap(self, C_train, C_test, pipeline, baseline, n_samples=2000):
        all_shap_values = []
        all_ratios = []
        n_trials, n_channels, _ = C_train.shape

        for run_index in range(C_test.shape[0]):
            current_run_cov = C_test[run_index]
            recorder = ShapRecorder()

            stable_predict_frozen = partial(
                self._stable_predict,
                current_run_cov=current_run_cov,
                reference_cov=baseline,
                pipeline=pipeline,
                recorder=recorder,
            )

            explainer = KernelExplainer(
                stable_predict_frozen, np.zeros((1, n_channels)), seed=run_index
            )
            shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)
            shap_values = np.asarray(shap_values).squeeze(axis=0)

            all_shap_values.append(shap_values)
            if recorder.ratios:
                all_ratios.append(np.concatenate(recorder.ratios, axis=0))

        return all_shap_values, all_ratios


class DeepShapleyValuesSPD:
    """Shapley values computed with KernelSHAP for deep SPD models."""

    def _stable_predict(self, mask_2d, current_run_cov, reference_cov, model):
        C_reconstructed = reconstruct_shap_samples(mask_2d, current_run_cov, reference_cov)
        C_tensor = torch.from_numpy(C_reconstructed).to(torch.float32)

        model.eval()
        with torch.no_grad():
            out = torch.softmax(model(C_tensor), dim=1)
        return out.detach().cpu().numpy()[:, 0]

    def KernelShapDeep(self, C_test, model, baseline, n_samples=2000):
        C_test_np = C_test.detach().cpu().numpy() if torch.is_tensor(C_test) else C_test
        n_trials, n_channels, _ = C_test_np.shape
        all_shap_values = []

        for run_index in range(C_test_np.shape[0]):
            current_run_cov = C_test_np[run_index]

            stable_predict_frozen = partial(
                self._stable_predict,
                current_run_cov=current_run_cov,
                reference_cov=baseline,
                model=model,
            )
            explainer = KernelExplainer(
                stable_predict_frozen, np.zeros((1, n_channels)), seed=run_index
            )
            shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)
            all_shap_values.append(shap_values)

        return all_shap_values

    def fit(self, X, y, n_splits=10, model_trained=None, model_config=None, model_type=None,
            baseline=None, n_samples=2000, epochs=200, lr=1e-3,
            X_train=None, y_train=None):
        """Compute Shapley values for a deep model on covariance matrices."""
        X = _as_covariances(X).astype(np.float32, copy=False)
        y = np.asarray(y)

        if model_trained is not None:
            if n_splits != 10:
                warnings.warn("`n_splits` is ignored for a pre-trained model")

            model_trained = model_trained.cpu().float()
            C_tensor = torch.from_numpy(X).to(torch.float32)
            encoder = LabelEncoder()
            y_tensor = torch.from_numpy(encoder.fit_transform(y)).long()
            self.classes_ = encoder.classes_
            if baseline is None:
                reference_X = X if X_train is None else _as_covariances(X_train)
                reference_y = y if y_train is None else np.asarray(y_train)
                split_baseline = _compute_cov_baseline(reference_X, reference_y)
            else:
                split_baseline = baseline

            shap_values = self.KernelShapDeep(
                C_tensor, model_trained, split_baseline, n_samples
            )
            model_trained.eval()
            with torch.no_grad():
                score = (model_trained(C_tensor).argmax(1) == y_tensor).float().mean().item()

            self.shap_values_ = [shap_values]
            self.scores_ = np.array([score])

        else:
            if model_config is None:
                raise ValueError(
                    "Give `model_trained`, or `model_config` to train a deep SPD model."
                )

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.classes_ = le.classes_
            model_class = SPDNetBatchNorm if model_type is None else model_type

            all_shap_values, all_scores = [], []
            for i in range(n_splits):
                torch.manual_seed(i)
                C_train, C_test, y_train_split, y_test = train_test_split(
                    X, y_encoded, train_size=0.8, stratify=y_encoded, random_state=i
                )

                C_train_t = torch.from_numpy(C_train).to(torch.float32)
                C_test_t = torch.from_numpy(C_test).to(torch.float32)
                y_train_t = torch.from_numpy(y_train_split).long()
                y_test_t = torch.from_numpy(y_test).long()

                split_baseline = (
                    baseline if baseline is not None
                    else _compute_cov_baseline(C_train, y_train_split)
                )

                new_model = model_class(**model_config, input_type="cov").float()
                train_model(
                    new_model, C_train_t, y_train_t, C_test_t, y_test_t,
                    epochs=epochs, lr=lr
                )
                new_model.eval()

                shap_values = self.KernelShapDeep(
                    C_test_t, new_model, split_baseline, n_samples
                )
                with torch.no_grad():
                    score = (new_model(C_test_t).argmax(1) == y_test_t).float().mean().item()

                all_shap_values.append(shap_values)
                all_scores.append(score)

            self.shap_values_ = all_shap_values
            self.scores_ = np.array(all_scores)

        return self
