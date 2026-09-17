import torch 
import numpy as np
import torch.nn as nn
#from spd_learn.models import SPDNet
from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDBatchNormMeanVar
from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
from moabb.paradigms import FilterBankMotorImagery
from spd_learn.functional import covariance
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from warnings import warn
import mne

np.random.seed(3)

montage = mne.channels.make_standard_montage('standard_1020')

# 2. Récupérer les positions sous forme de dictionnaire {nom: [x, y, z]}
positions = montage.get_positions()['ch_pos']

class SPDNetBatchNorm(nn.Module):

    def __init__(
        self,
        input_type="raw",
        cov_method=covariance,
        subspacedim=None,
        threshold=1e-4,
        upper=True,
        n_chans=None,
        n_outputs=None,
        bn = None
    ):
        super().__init__()

        if subspacedim is None:
            warn(
                "subspacedim is None, using the default value of "
                "the number of channels",
                UserWarning,
            )
            subspacedim = n_chans

        if input_type == "raw":
            self.cov = CovLayer(method=cov_method)
        elif input_type == "cov":
            self.cov = nn.Identity()

        self.bimap = BiMap(n_chans, subspacedim)
        self.bn = bn
        self.reeig = ReEig(threshold)
        self.logeig = LogEig(upper=upper)
        self.len_last_layer = (
            subspacedim * (subspacedim + 1) // 2 if upper else subspacedim**2
        )
        self.classifier = nn.Linear(self.len_last_layer, n_outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        X = self.cov(X)
        X = self.bimap(X)
        if self.bn is not None:
            X = self.bn(X)
        X = self.reeig(X)
        X = self.logeig(X)
        X = self.classifier(X)

        return X


def train_model(model, X_train, y_train, X_test, y_test, epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Test Acc: {acc:.4f}")

def permute_channel_across_times(X, y, model, feature, random_state=None): 
    rng = np.random.default_rng(random_state) 
    X_modified = X.clone() 

    for trial in range(X.shape[0]): 
        perm = rng.permutation(X.shape[2]) 
        X_modified[trial, feature, :] = X_modified[trial, feature, perm] 
    
    model.eval()
    with torch.no_grad():
        perturbed_acc = (model(X_modified).argmax(1) == y).float().mean().item()
    return perturbed_acc

def _single_iteration(X_data, y_data, model_config, method, n_perm, seed=None):
    if seed is not None:

        np.random.seed(seed)

        torch.manual_seed(seed)

        torch.use_deterministic_algorithms(True, warn_only=True)
        


    rng = np.random.default_rng(seed)

    # Split Train/Test
    train_eeg, test_eeg, train_y, test_y = train_test_split(
        X_data, y_data, stratify=y_data, test_size=0.2, random_state=seed
    )
    
    # Conversion Tensors
    train_eeg = torch.from_numpy(train_eeg).to(torch.float64)
    test_eeg = torch.from_numpy(test_eeg).to(torch.float64)
    train_y = torch.from_numpy(train_y).long()
    test_y = torch.from_numpy(test_y).long()
    
    # Init & Train
    model = SPDNetBatchNorm(**model_config).double()
    train_model(model, train_eeg, train_y, test_eeg, test_y)
    
    model.eval()
    with torch.no_grad():
        baseline = (model(test_eeg).argmax(1) == test_y).float().mean().item()

    print(f"  Baseline Accuracy: {baseline:.2f}")
    
    feature_importance = np.zeros(X_data.shape[1])
    for feature in range(X_data.shape[1]):
        scores = []
        for k in range(n_perm):
            perm_seed = rng.integers(0, 1_000_000)
            score = permute_channel_across_times(test_eeg, test_y, model, feature, perm_seed)
            scores.append(score)
        
        feature_importance[feature] = baseline - np.mean(scores)
        
    return baseline, feature_importance

def fit(X, y, meta, model_config, n_iter=2, n_perm=2, method="across_times"):
    results_dic = {method: {}}
    
    subjects = np.unique(meta.subject)
    
    for subject in subjects:
        print(f"\nProcessing Subject: {subject}")
        subject_mask = (meta.subject == subject)
        data_subj = X[subject_mask]
        y_subj = y[subject_mask]

        subject_results = []
        for i in range(n_iter):
            print(f" Iteration {i+1}/{n_iter}")
            res = _single_iteration(data_subj, y_subj, model_config, method, n_perm, seed=i)
            subject_results.append(res)

        accuracies = np.array([r[0] for r in subject_results])
        importances = np.array([r[1] for r in subject_results])

        results_dic[method][subject] = {
            "accuracy": accuracies,
            "importance": importances
        }
    
    return results_dic



