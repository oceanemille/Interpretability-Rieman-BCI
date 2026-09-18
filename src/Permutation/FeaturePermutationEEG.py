import warnings
import numpy as np
import torch
import torch.nn as nn
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyriemann.utils.distance import distance_riemann
from sklearn.base import clone
from joblib import Parallel, delayed


class FeaturePermutationEEG:
    """Permutation feature importance for a pyRiemann classifier.

    Two modes are available through ``fit()``:
    - With ``pipeline_pretrained``, ``X`` and ``y`` are raw EEG test data with
      shape ``(n_trials, n_channels, n_times)`` and their labels.
      ``n_splits`` is ignored.
    - With ``pipeline``, ``X`` and ``y`` contain the complete dataset. The
      pipeline is trained and evaluated on ``n_splits`` reproducible splits.
    """

    def fit(self, X, y, n_splits=10, n_perm=10,
            pipeline=None, pipeline_pretrained=None, n_jobs = -1):
        if pipeline_pretrained is not None:
            if n_splits != 10:
                warnings.warn(
                    "`n_splits` is ignored when a pre-trained pipeline is "
                    "provided because no repeated train/test split is needed."
                )
            baseline, importance = self._compute_importance(
                pipeline_pretrained, X, y, n_perm=n_perm
            )
            self.accuracy_ = np.array([baseline])
            self.importance_ = np.array([importance])

        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._single_iteration)(X, y, pipeline, n_perm, seed=i)
                for i in range(n_splits))
            
            self.accuracy_ = np.array([r[0] for r in results])
            self.importance_ = np.array([r[1] for r in results])

        return self

    def _single_iteration(self, X, y, pipeline_template,n_perm, seed=None):
        train_eeg, test_eeg, train_y, test_y = train_test_split(
            X, y, stratify=y, random_state=seed, test_size=0.2
        )

        pipeline = clone(pipeline_template)
        pipeline.fit(train_eeg, train_y)

        return self._compute_importance(
            pipeline,test_eeg, test_y,
            n_perm=n_perm, seed=seed
        )

    def _compute_importance(self, pipeline, X_test, y_test,
                             n_perm, seed=None):
        """Compute baseline accuracy and importance for a fitted classifier."""
        baseline = np.mean(pipeline.predict(X_test) == y_test)

        n_features = X_test.shape[1]
        feature_importance = np.zeros(n_features)

        for feature in range(n_features):
            mean_perturbed = self._permute_multiple_times(
                X_test, y_test, pipeline, feature, n_perm,
                seed
            )
            feature_importance[feature] = baseline - mean_perturbed

        return baseline, feature_importance

    def _permute_multiple_times(self, X, y_true, pipeline, feature, n_perm, seed, can_compute_ratio=True):
        rng = np.random.default_rng(seed)
        scores = []

        for _ in range(n_perm):
            perm_seed = rng.integers(0, 1_000_000)
            score = self.permute_channel_across_times(
                X, y_true, pipeline, feature, random_state=perm_seed
            )

            scores.append(score)

        return np.mean(scores)


    def permute_channel_across_times(self, X, y_true, pipeline,
                                      feature, random_state=None):
        rng = np.random.default_rng(random_state)
        X_modified = X.copy()
        for trial in range(X.shape[0]):
            perm = rng.permutation(X.shape[2])
            X_modified[trial, feature, :] = X_modified[trial, feature, perm]

        perturbed = np.mean(pipeline.predict(X_modified) == y_true)
        return perturbed


class DeepFeaturePermutationEEG:
    """Permutation feature importance for a torch model such as SPDNet.

    Two modes are available through ``fit()``:
    - With ``model``, ``X`` and ``y`` must be test tensors. Labels must already
      be encoded as integers from 0 to ``n_classes - 1``. ``n_splits`` is
      ignored.
    - With ``model_config`` and ``model_type``, ``X`` and ``y`` are NumPy
      arrays containing the complete dataset. Labels are encoded automatically
      and training is repeated over ``n_splits`` splits.

    Only channel-wise permutation across time samples is currently supported.
    """

    def fit(self, X, y, n_splits=10, n_perm=10, seed=None,
            model=None, model_config=None, model_type=None, n_jobs = -1):

        if model is not None:
            if n_splits != 10:
                warnings.warn(
                    "`n_splits` is ignored when a pre-trained model is "
                    "provided because no repeated split or training is needed."
                )
            baseline, importance = self._compute_importance(
                model, X, y, n_perm=n_perm, seed=seed
            )
            self.accuracy_ = np.array([baseline])
            self.importance_ = np.array([importance])

        else:
            if model_config is None or model_type is None:
                raise ValueError(
                    "Provide either `model` with test tensors, or both "
                    "`model_type` and `model_config` to train a new model."
                )

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.classes_ = le.classes_

            results = [self._train_and_compute_importance(
                    X, y_encoded, model_config, model_type, n_perm=n_perm, seed=i
                )
                for i in range(n_splits)]
            
            self.accuracy_ = np.array([r[0] for r in results])
            self.importance_ = np.array([r[1] for r in results])

        return self

    def _compute_importance(self, model, X_test, y_test, n_perm, seed=None):
        """Compute baseline accuracy and importance for an in-memory model."""
        rng = np.random.default_rng(seed)
        model.eval()
        with torch.no_grad():
            baseline = (model(X_test).argmax(1) == y_test).float().mean().item()

        n_features = X_test.shape[1]
        feature_importance = np.zeros(n_features)

        for feature in range(n_features):
            scores = []
            for _ in range(n_perm):
                perm_seed = rng.integers(0, 1_000_000)
                scores.append(
                    self._permute_channel_across_times(
                        X_test, y_test, model, feature, perm_seed
                    )
                )
            feature_importance[feature] = baseline - np.mean(scores)

        return baseline, feature_importance

    def _train_and_compute_importance(self, X_data, y_data, model_config,
                                       model_type, n_perm, seed=None):
        """Train a model and compute its permutation importance."""

        if seed is not None:
            torch.manual_seed(seed)
        train_eeg, test_eeg, train_y, test_y = train_test_split(
            X_data, y_data, stratify=y_data, test_size=0.2, random_state=seed
        )
        train_eeg = torch.from_numpy(train_eeg).to(torch.float32)
        test_eeg = torch.from_numpy(test_eeg).to(torch.float32)
        train_y = torch.from_numpy(train_y).long()
        test_y = torch.from_numpy(test_y).long()

        model = model_type(**model_config).float()
        self.train_model(model, train_eeg, train_y, test_eeg, test_y)

        return self._compute_importance(model, test_eeg, test_y, n_perm, seed=seed)

    @staticmethod
    def train_model(model, X_train, y_train, X_test, y_test,
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

    @staticmethod
    def _permute_channel_across_times(X, y_true, model, feature, random_state=None):
        rng = np.random.default_rng(random_state)
        X_modified = X.clone()
        for trial in range(X.shape[0]):
            perm = rng.permutation(X.shape[2])
            X_modified[trial, feature, :] = X_modified[trial, feature, perm]

        model.eval()
        with torch.no_grad():
            perturbed_acc = (model(X_modified).argmax(1) == y_true).float().mean().item()
        return perturbed_acc
