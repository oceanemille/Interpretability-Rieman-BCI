from .FeaturePermutationEEG import FeaturePermutationEEG, DeepFeaturePermutationEEG
from .FeaturePermutationSPD import FeaturePermutationSPD, DeepFeaturePermutationSPD
import torch 

def run_multi_subjects_spd(data: dict, n_splits=10, n_perm=10, model_trained = None, classifier = None, n_jobs = -1):
    """
    Calcule l'importance sur plusieurs sujets

    ParametersParallel(n_jobs=n_jobs)
    ----------
    data : dict {subject_name: (X, y)}
        Typiquement obtenu via `load_from_moabb(...)`, mais peut être
        construit à la main si vous avez plusieurs sujets/sessions.
    """
    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating importance for {subject_name}...")
        perm = FeaturePermutationSPD()
        perm.fit(X, y, n_splits=n_splits, model_trained = model_trained, classifier = classifier, n_jobs = n_jobs)
        results_dic[subject_name] = {
        "importance": perm.importance_,  
        "accuracy": perm.accuracy_,
    }

    return results_dic  
            

def run_multi_subjects_eeg(data: dict, n_splits=10, n_perm=10, pipeline = None, pipeline_pre_trained = None, n_jobs = -1):

    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating importance for {subject_name}...")
        perm = FeaturePermutationEEG()
        perm.fit(X, y, n_splits=n_splits, n_perm=n_perm, pipeline=pipeline, pipeline_pre_trained = pipeline_pre_trained, n_jobs = n_jobs)
        results_dic[subject_name] = {
        "importance": perm.importance_,  
        "accuracy": perm.accuracy_,
    }

    return results_dic



def run_multi_subjects_deep_spd(data: dict, n_splits=10, model_config= None, model = None, model_type = None, n_jobs = -1):
    """
    Calcule l'importance sur plusieurs sujets

    Parameters
    ----------
    data : dict {subject_name: (X, y)}
        Typiquement obtenu via `load_from_moabb(...)`, mais peut être
        construit à la main si vous avez plusieurs sujets/sessions.
    """
    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating importance for {subject_name}...")
        perm = DeepFeaturePermutationSPD()
        perm.fit(X, y, n_splits=n_splits, model_config=model_config, model=model, model_type=model_type, n_jobs = n_jobs)
        results_dic[subject_name] = {
        "importance": perm.importance_,  
        "accuracy": perm.accuracy_,
    }
    return results_dic

def run_multi_subjects_deep_eeg(data: dict, n_splits=10, n_perm=10, model_config= None, model = None, model_type = None, n_jobs = -1):
    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating importance for {subject_name}...")
        perm = DeepFeaturePermutationEEG()
        perm.fit(X, y, n_splits=n_splits, n_perm=n_perm, model_config=model_config, model=model, model_type=model_type, n_jobs = n_jobs)
        results_dic[subject_name] = {
        "importance": perm.importance_, 
        "accuracy": perm.accuracy_,
    }
    return results_dic
