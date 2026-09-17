import warnings
import numpy as np
import torch
import torch.nn as nn
from pyriemann.classification import MDM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyriemann.utils.distance import distance_riemann
from joblib import Parallel, delayed


def _ablate_channel(C, feature):
    """Met à zéro les termes off-diagonaux du canal `feature`, en gardant
    sa variance intacte.

    C : ndarray (n_trials, n_channels, n_channels), matrices de covariance.
    """
    C_ablated = C.copy()
    diag = C[:, feature, feature].copy()
    C_ablated[:, feature, :] = 0
    C_ablated[:, :, feature] = 0
    C_ablated[:, feature, feature] = diag
    return C_ablated


class FeaturePermutationSPD:
    """Importance des canaux par ablation des covariances croisées.

    Pour chaque canal, on met à zéro ses termes off-diagonaux dans les
    matrices de covariance (en conservant sa variance), et on mesure la
    chute d'accuracy par rapport à la baseline.

    IMPORTANT : `X` doit être des matrices de covariance SPD de forme
    (n_trials, n_channels, n_channels), pas de l'EEG brut. Si vous partez
    d'EEG brut, appliquez d'abord
    `pyriemann.estimation.Covariances().fit_transform(X_raw)`.

    Deux modes via `fit()` :
    - `model` fourni : classifieur déjà entraîné, `X, y` = données de test.
      `n_splits` est ignoré.
    - `classifier` fourni (ou rien, défaut MDM) : `X, y` = dataset complet,
      split + entraînement répétés `n_splits` fois.
    """

    def fit(self, X, y, n_splits=10, model_trained=None, classifier=None, n_jobs = -1):
        if model_trained is not None:
            if n_splits != 10:
                warnings.warn(
                    "`n_splits` est ignoré quand un modèle pré-entraîné est "
                    "fourni (pas de split train/test à répéter)."
                )
            baseline, importance = self._compute_importance(model_trained, X, y)
            self.accuracy_ = np.array([baseline])
            self.importance_ = np.array([importance])

        else:
            clf_class = classifier or MDM
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._single_iteration)(X, y, clf_class, seed=i)
                for i in range(n_splits)
                )
            self.accuracy_ = np.array([r[0] for r in results])
            self.importance_ = np.array([r[1] for r in results])

        return self

    def _single_iteration(self, X, y, clf_class, seed=None):
        C_train, C_test, train_y, test_y = train_test_split(
            X, y, stratify=y, random_state=seed, test_size=0.2
        )

        clf = clf_class()
        clf.fit(C_train, train_y)        

        return self._compute_importance(clf, C_test, test_y)

    def _compute_importance(self, clf, C_test, y_test):
        """Baseline + importance par ablation, pour un classifieur entraîné."""
        baseline = np.mean(clf.predict(C_test) == y_test)

        n_features = C_test.shape[1]
        feature_importance = np.zeros(n_features)

        for feature in range(n_features):
            C_ablated = _ablate_channel(C_test, feature) 
            perturbed_score = np.mean(clf.predict(C_ablated) == y_test)
            feature_importance[feature] = baseline - perturbed_score

        return baseline, feature_importance

class DeepFeaturePermutationSPD:
    """Importance des canaux par ablation, pour un modèle torch (SPDNet).

    `X` doit être des matrices de covariance SPD
    (n_trials, n_channels, n_channels).

    Deux modes via `fit()` :
    - `model` fourni : modèle déjà entraîné, `X, y` = TENSORS torch de test
      (labels déjà encodés 0..n_classes-1). `n_splits` est ignoré.
    - `model_type` + `model_config` fournis : `X, y` = ndarray numpy
      (dataset complet), entraînement répété `n_splits` fois.
    """

    def fit(self, X, y, n_splits=10, model=None, model_config=None, model_type=None, n_jobs = -1):
        if model is not None:
            if n_splits != 10:
                warnings.warn(
                    "`n_splits` est ignoré quand un modèle pré-entraîné est "
                    "fourni (pas de split/entraînement à répéter)."
                )
            baseline, importance = self._compute_importance(model, X, y)
            self.accuracy_ = np.array([baseline])
            self.importance_ = np.array([importance])

        else:
            if model_config is None or model_type is None:
                raise ValueError(
                    "Fournissez soit `model` (déjà entraîné, X/y en tensors "
                    "de test), soit `model_type` ET `model_config` "
                    "(classe du modèle + ses hyperparamètres) pour entraîner."
                )

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.classes_ = le.classes_

            results = [self._train_and_compute_importance(
                    X, y_encoded, model_config, model_type, seed=i
                )
                for i in range(n_splits)]
            
            self.accuracy_ = np.array([r[0] for r in results])
            self.importance_ = np.array([r[1] for r in results])

        return self

    def _compute_importance(self, model, C_test, y_test):
        """Baseline + importance par ablation, pour un modèle torch entraîné."""
        model.eval()
        with torch.no_grad():
            baseline = (model(C_test).argmax(1) == y_test).float().mean().item()

        n_features = C_test.shape[1]
        feature_importance = np.zeros(n_features)

        for feature in range(n_features):
            C_ablated = C_test.clone()
            diag = C_test[:, feature, feature].clone()
            C_ablated[:, feature, :] = 0
            C_ablated[:, :, feature] = 0
            C_ablated[:, feature, feature] = diag

            with torch.no_grad():
                perturbed = (model(C_ablated).argmax(1) == y_test).float().mean().item()
            feature_importance[feature] = baseline - perturbed

        return baseline, feature_importance

    def _train_and_compute_importance(self, X_data, y_data, model_config,
                                       model_type, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        C_train, C_test, train_y, test_y = train_test_split(
            X_data, y_data, stratify=y_data, test_size=0.2, random_state=seed
        )
        C_train = torch.from_numpy(C_train).to(torch.float32)
        C_test = torch.from_numpy(C_test).to(torch.float32)
        train_y = torch.from_numpy(train_y).long()
        test_y = torch.from_numpy(test_y).long()

        model = model_type(**model_config, input_type = "cov").float()
        self.train_model(model, C_train, train_y, C_test, test_y)

        return self._compute_importance(model, C_test, test_y)

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