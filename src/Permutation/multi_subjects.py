from .FeaturePermutationEEG import FeaturePermutationEEG, DeepFeaturePermutationEEG
from .FeaturePermutationSPD import FeaturePermutationSPD, DeepFeaturePermutationSPD

def compute_pfi_spd_for_subjects(data: dict, n_splits=10, n_perm=10, model_trained = None, classifier = None, n_jobs = -1):
    """
    Compute importance for multiple subjects

    ParametersParallel(n_jobs=n_jobs)
    ----------
    data : dict {subject_name: (X, y)}
        Typically returned by ``load_from_moabb(...)``. It can also be built
        manually when several subjects or sessions are available.
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
            

def compute_pfi_eeg_for_subjects(data: dict, n_splits=10, n_perm=10, pipeline = None, pipeline_pretrained = None, n_jobs = -1):

    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating importance for {subject_name}...")
        perm = FeaturePermutationEEG()
        perm.fit(X, y, n_splits=n_splits, n_perm=n_perm, pipeline=pipeline, pipeline_pretrained = pipeline_pretrained, n_jobs = n_jobs)
        results_dic[subject_name] = {
        "importance": perm.importance_,  
        "accuracy": perm.accuracy_,
    }

    return results_dic



def compute_deep_pfi_spd_for_subjects(data: dict, n_splits=10, model_config= None, model = None, model_type = None, n_jobs = -1):
    """
    Compute importance for multiple subjects

    Parameters
    ----------
    data : dict {subject_name: (X, y)}
        Typically returned by ``load_from_moabb(...)``. It can also be built
        manually when several subjects or sessions are available.
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

def compute_deep_pfi_eeg_for_subjects(data: dict, n_splits=10, n_perm=10, model_config= None, model = None, model_type = None, n_jobs = -1):
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


# Backward-compatible aliases for existing scripts.
run_multi_subjects_spd = compute_pfi_spd_for_subjects
run_multi_subjects_eeg = compute_pfi_eeg_for_subjects
run_multi_subjects_deep_spd = compute_deep_pfi_spd_for_subjects
run_multi_subjects_deep_eeg = compute_deep_pfi_eeg_for_subjects
