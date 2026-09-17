import warnings
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from shap import KernelExplainer
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann


def reconstruct_shap_samples(mask_2d, current_run_signal, reference_signal):
    """Reconstruit les signaux EEG définis par les masques SHAP."""
    n_simulations, n_channels = mask_2d.shape
    active = mask_2d > 0.5

    X_reconstructed = np.where(
        active[:, :, None],
        current_run_signal[None, :, :],
        reference_signal[None, :, :],
    ).astype(current_run_signal.dtype)

    return X_reconstructed


class ShapRecorder:
    def __init__(self):
        self.masks = []
        self.ratios = []
        self.predictions = []

    def add(self, masks, ratios, predictions):
        self.masks.append(masks.copy())
        self.ratios.append(ratios.copy())
        self.predictions.append(predictions.copy())


class ShapleyValuesEEG:
    """Importance des canaux par valeurs de Shapley (KernelSHAP), pour une
    pipeline pyriemann/sklearn (ex: Covariances + MDM)."""

    def fit(self, X, y, n_splits=10, pipeline=None, pipeline_pretrained=None, baseline=None, n_samples=1000):
        """Calcule les valeurs de Shapley d'une pipeline sur des signaux EEG."""
        if pipeline_pretrained is not None:
            if n_splits != 10:
                warnings.warn("`n_splits` est ignoré avec un modèle pré-entraîné.")
            baseline_signal = baseline if baseline is not None else self._compute_noise_baseline(X, y)
            shap_values, ratios = self.KernelShap(X, pipeline_pretrained, baseline_signal, n_samples)
            score = np.mean(pipeline_pretrained.predict(X) == y)
            self.shap_values_ = [shap_values]
            self.ratios_ = [ratios]
            self.scores_ = np.array([score])

        else:
            if pipeline is None:
                raise ValueError("Fournissez soit `pipeline_pre_trained` (déjà entraîné), soit `pipeline` (à entraîner).")

            all_shap_values, all_ratios, all_scores = [], [], []
            for i in range(n_splits):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=0.8, stratify=y, random_state=i
                )
                clf = clone(pipeline)
                clf.fit(X_train, y_train)

                split_baseline = baseline if baseline is not None else self._compute_noise_baseline(X_train, y_train)
                shap_values, ratios = self.KernelShap(X_test, clf, split_baseline, n_samples)

                all_shap_values.append(shap_values)
                all_ratios.append(ratios)
                all_scores.append(clf.score(X_test, y_test))

            self.shap_values_ = all_shap_values
            self.ratios_ = all_ratios
            self.scores_ = np.array(all_scores)

        return self

    def _stable_predict(self, mask_2d, current_run_signal, reference_signal,
                         pipeline, recorder=None):
        X_reconstructed = reconstruct_shap_samples(mask_2d, current_run_signal, reference_signal)
        predictions = pipeline.predict_proba(X_reconstructed)[:, 0]

        if recorder is not None and hasattr(pipeline[-1], "covmeans_"):
            covs = pipeline[0].transform(X_reconstructed)
            C0, C1 = pipeline[-1].covmeans_[0], pipeline[-1].covmeans_[1]
            d0 = np.array([distance_riemann(c, C0) for c in covs])
            d1 = np.array([distance_riemann(c, C1) for c in covs])
            dc = distance_riemann(C0, C1)
            ratios = (d0 - d1) / dc
            recorder.add(mask_2d, ratios, predictions)

        return predictions

    def _compute_noise_baseline(self, X_train, y_train):
        """Baseline = signal moyen + bruit dont la variance vient des
        covariances moyennes par classe (générique, pas limité à 2 classes)."""
        covariance = Covariances()
        covs = covariance.transform(X_train)
        classes = np.unique(y_train)
        class_means = [mean_riemann(covs[y_train == c]) for c in classes]
        var_baseline = np.diag(np.mean(class_means, axis=0))

        mean_signal = np.mean(X_train, axis=0)
        noise = np.random.normal(
            0, scale=np.sqrt(var_baseline)[:, np.newaxis], size=mean_signal.shape
        )
        return mean_signal + noise

    def KernelShap(self, X_test, pipeline, baseline, n_samples=2000):
        all_shap_values = []
        all_ratios = []
        n_trials, n_channels, n_times = X_test.shape

        for run_index in range(X_test.shape[0]):
            current_run_signal = X_test[run_index]
            recorder = ShapRecorder()

            stable_predict_frozen = partial(
                self._stable_predict,
                current_run_signal=current_run_signal,
                reference_signal=baseline,
                pipeline=pipeline,
                recorder=recorder,
            )

            explainer = KernelExplainer(
                stable_predict_frozen, np.zeros((1, n_channels)), seed=run_index
            )
            shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)

            all_shap_values.append(shap_values)
            if recorder.ratios:
                all_ratios.append(np.concatenate(recorder.ratios, axis=0))

        return all_shap_values, all_ratios


class DeepShapleyValuesEEG:
    """Importance des canaux par valeurs de Shapley (KernelSHAP), pour un
    modèle torch (SPDNet)."""

    def train_model(self, model, X_train, y_train, X_test, y_test,
                     epochs=200, lr=1e-3, verbose=True):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    acc = (model(X_test).argmax(1) == y_test).float().mean().item()
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Test Acc: {acc:.4f}")

        return model

    def _stable_predict(self, mask_2d, current_run_signal, reference_signal, model):
        X_reconstructed = reconstruct_shap_samples(mask_2d, current_run_signal, reference_signal)
        X_tensor = torch.from_numpy(X_reconstructed).to(torch.float32)

        model.eval()
        with torch.no_grad():
            out = torch.softmax(model(X_tensor), dim=1)
        return out.detach().cpu().numpy()

    def _compute_noise_baseline(self, X_train, y_train):
        return ShapleyValuesEEG()._compute_noise_baseline(X_train, y_train)

    def KernelShapDeep(self, X_test, model, baseline, n_samples=2000):
        X_test_np = X_test.detach().cpu().numpy() if torch.is_tensor(X_test) else X_test
        n_trials, n_channels, n_times = X_test_np.shape
        all_shap_values = []

        for run_index in range(X_test_np.shape[0]):
            current_run_signal = X_test_np[run_index]

            stable_predict_frozen = partial(
                self._stable_predict,
                current_run_signal=current_run_signal,
                reference_signal=baseline,
                model=model,
            )
            explainer = KernelExplainer(
                stable_predict_frozen, np.zeros((1, n_channels)), seed=run_index
            )
            shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)
            all_shap_values.append(shap_values)

        return all_shap_values

    def fit(self, X, y, n_splits=10, model=None, model_config=None, model_type=None,
            baseline=None, n_samples=2000, epochs=200, lr=1e-3):

        if model is not None:
            if n_splits != 10:
                warnings.warn("`n_splits` est ignoré avec un modèle pré-entraîné.")
            X_numpy = X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)
            y_numpy = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            X_tensor = torch.from_numpy(X_numpy).to(torch.float32)
            encoder = LabelEncoder()
            y_tensor = torch.from_numpy(encoder.fit_transform(y_numpy)).long()
            self.classes_ = encoder.classes_
            model = model.cpu().float()
            baseline_signal = (
                baseline if baseline is not None
                else self._compute_noise_baseline(X_numpy, y_numpy)
            )
            shap_values = self.KernelShapDeep(X_tensor, model, baseline_signal, n_samples)
            model.eval()
            with torch.no_grad():
                score = (model(X_tensor).argmax(1) == y_tensor).float().mean().item()
            self.shap_values_ = [shap_values]
            self.scores_ = np.array([score])

        else:
            if model_config is None or model_type is None:
                raise ValueError("Fournissez `model`, ou `model_type` + `model_config` pour entraîner.")

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.classes_ = le.classes_

            all_shap_values, all_scores = [], []
            for i in range(n_splits):
                torch.manual_seed(i)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, train_size=0.8, stratify=y_encoded, random_state=i
                )
                X_train_t = torch.from_numpy(X_train).to(torch.float32)
                X_test_t = torch.from_numpy(X_test).to(torch.float32)
                y_train_t = torch.from_numpy(y_train).long()
                y_test_t = torch.from_numpy(y_test).long()

                new_model = model_type(**model_config).float()
                self.train_model(new_model, X_train_t, y_train_t, X_test_t, y_test_t, epochs=epochs, lr=lr)
                new_model.eval()

                split_baseline = (
                    baseline if baseline is not None
                    else self._compute_noise_baseline(X_train, y_train)
                )
                shap_values = self.KernelShapDeep(X_test_t, new_model, split_baseline, n_samples)
                score = (new_model(X_test_t).argmax(1) == y_test_t).float().mean().item()

                all_shap_values.append(shap_values)
                all_scores.append(score)

            self.shap_values_ = all_shap_values
            self.scores_ = np.array(all_scores)

        return self