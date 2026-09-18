import numpy as np
import torch
import shap
from shap import KernelExplainer
from functools import partial
from sklearn.model_selection import train_test_split
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.covariance import normalize
from spd_learn.functional import covariance
from spd_learn.modules import CovLayer
from src.SPDNet.SPDNet import SPDNetBatchNorm, train_model
from pyriemann.utils.distance import distance_riemann


class ShapRecorder:
    def __init__(self):
        self.masks = []
        self.ratios = []
        self.predictions = []

    def add(self, masks, ratios, predictions):
        self.masks.append(masks.copy())
        self.ratios.append(ratios.copy())
        self.predictions.append(predictions.copy())


def stable_predict(mask_2d,n_channels,current_run_signal,reference_signal,clf, deep=False, recorder=None):
    if deep:
        model= clf
            # mask_2d has shape (n_simulations, n_channels).
        n_simulations = mask_2d.shape[0]
        X_reconstructed = np.array([current_run_signal for _ in range(n_simulations)])
        
        for i in range(n_simulations):
            for ch in range(n_channels):
                if mask_2d[i, ch] < 0.5:
                    X_reconstructed[i, ch, :] = reference_signal[ch, :]
                    X_reconstructed[i,:,ch] = reference_signal[:, ch]
                    
        X_reconstructed = torch.tensor(
        X_reconstructed,
        dtype=torch.float32

    )
        with torch.no_grad():
            out = torch.softmax(model(X_reconstructed), dim=1)
        
        return out.detach().numpy()[:,0]

    else : 

        # mask_2d has shape (n_simulations, n_channels).
        n_simulations = mask_2d.shape[0]
        X_reconstructed = np.array([current_run_signal for _ in range(n_simulations)])
        
        for i in range(n_simulations):
            for ch in range(n_channels):
                # A SHAP mask value of 1 selects the current trial.
                if mask_2d[i, ch] < 0.5:
                    X_reconstructed[i, ch, :] = reference_signal[ch, :]
                    X_reconstructed[i,:,ch] = reference_signal[:, ch]

    predictions = clf.predict_proba(X_reconstructed)[:,0]

    if recorder is not None:

        C0 = clf.covmeans_[0]
        C1 = clf.covmeans_[1]

        d0 = np.array([distance_riemann(cov, C0) for cov in X_reconstructed])
        d1 = np.array([distance_riemann(cov, C1) for cov in X_reconstructed])
        dc = distance_riemann(C0, C1)

        ratios = (d0 - d1) / dc

        recorder.add(
            mask_2d,
            ratios,
            predictions
        )


    return predictions


def KernelShap(C_train,C_test, clf, baseline,n_samples=2000):
    all_shap_values = []
    all_ratios = []
    n_trials, n_channels, _ = C_train.shape
    reference_signal = baseline

    for run_index in range(C_test.shape[0]):
        current_run_signal = C_test[run_index] 
        recorder = ShapRecorder()
        
        stable_predict_frozen = partial(stable_predict, n_channels=n_channels, 
                                        current_run_signal=current_run_signal, 
                                        reference_signal=reference_signal, clf=clf,recorder=recorder)
        explainer = KernelExplainer(stable_predict_frozen, np.zeros((1, n_channels)),seed=run_index)
        
        
        shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)

        all_shap_values.append(shap_values)
        trial_ratios = np.concatenate(recorder.ratios, axis=0)
        all_ratios.append(trial_ratios)

    return all_shap_values, all_ratios

def KernelShapDeep(X_train,X_test,model,baseline, n_samples=2000):
    X_train = X_train.detach().numpy()
    X_test  = X_test.detach().numpy()
    n_trials, n_channels,_ = X_train.shape
    reference_signal = baseline

    all_shap_values = []
    for run_index in range(X_test.shape[0]):
        current_run_signal = X_test[run_index] 
        
        stable_predict_frozen = partial(stable_predict, n_channels=n_channels, 
                                        current_run_signal=current_run_signal, 
                                        reference_signal=reference_signal, clf=model, deep=True)
        explainer = KernelExplainer(stable_predict_frozen, np.zeros((1, n_channels)),seed=run_index)
        
        shap_values = explainer.shap_values(np.ones((1, n_channels)), nsamples=n_samples)

        all_shap_values.append(shap_values)

    return all_shap_values


def compute_shapley(X,y, n_splits, clf, deep=False):
    if deep :
        model_config = clf
        all_shap_values = []
        all_scores = []
        all_ratios = []

        for i in range(n_splits):
            print("Split", i)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, stratify=y, random_state=i
            )

            covs = Covariances()

            C_train = covs.fit_transform(X_train)
            C_test = covs.transform(X_test)
            baseline = np.diag(np.diag(mean_riemann(C_train)))

            C_train = torch.tensor(C_train).float()
            C_test = torch.tensor(C_test).float()
            baseline = torch.tensor(baseline).float()


            model = SPDNetBatchNorm(**model_config, input_type='cov').float()
            train_model(model, C_train, y_train, C_test, y_test, epochs=200, lr=1e-3)
            model.eval()
 
            shap_values, ratios = KernelShapDeep(C_train, C_test, model, baseline, n_samples=1000)
            all_shap_values.append(shap_values)
            all_scores.append((model(C_test).argmax(1) == y_test).float().mean().item())
            all_ratios.append(ratios)

    else : 
        all_shap_values = []
        all_scores = []
        all_ratios = []
        for i in range (n_splits):
            print("Split",i)
            X_train,X_test,y_train,y_test = train_test_split(X,y, train_size=0.8, stratify=y, random_state=i)
            covs_transformer = Covariances()
            C_train = covs_transformer.fit_transform(X_train) 
            covs_right = C_train[y_train=='right_hand']
            covs_left = C_train[y_train=='left_hand']
            var_right = np.array([[covs_right[j,i,i] for i in range (covs_right.shape[1])] for j in range(covs_right.shape[0])])
            var_left = np.array([[covs_left[j,i,i] for i in range (covs_left.shape[1])] for j in range(covs_left.shape[0])])
            baseline = np.diag(np.diag((var_left + var_right) / 2))
            C_test = covs_transformer.transform(X_test)
            clf.fit(C_train,y_train)
            probas = clf.predict_proba(C_test)
            shap_values, ratios = KernelShap(C_train,C_test, clf, baseline = baseline, n_samples=1000)
            all_shap_values.append(shap_values)
            all_scores.append(clf.score(C_test,y_test))
            all_ratios.append(ratios)

    return all_shap_values, all_scores, all_ratios


