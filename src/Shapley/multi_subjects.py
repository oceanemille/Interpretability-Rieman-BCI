from .ShapleyValuesEEG import ShapleyValuesEEG, DeepShapleyValuesEEG
from .ShapleyValuesSPD import ShapleyValuesSPD, DeepShapleyValuesSPD


def compute_kernel_shap_spd_for_subjects(data, n_splits=10, model_trained = None, classifier = None):
    """Compute KernelSHAP values from SPD data for multiple subjects."""
    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating shap_values for {subject_name}...")
        perm = ShapleyValuesSPD()
        perm.fit(X, y, n_splits=n_splits, model_trained=model_trained, model=classifier)
        results_dic[subject_name] = {
        "importance": perm.shap_values_,
        "accuracy": perm.scores_,
    }

    return results_dic  
            

def compute_kernel_shap_eeg_for_subjects(data, n_splits=10, n_perm=10, pipeline = None, pipeline_pre_trained = None, n_jobs = -1):

    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating shap_values for {subject_name}...")
        perm = ShapleyValuesEEG()
        perm.fit(X, y, n_splits=n_splits, pipeline=pipeline, pipeline_pretrained=pipeline_pre_trained)
        results_dic[subject_name] = {
        "importance": perm.shap_values_,
        "accuracy": perm.scores_,
    }

    return results_dic



def compute_deep_kernel_shap_spd_for_subjects(data, n_splits=10, model_config= None, model = None, model_type = None):
    """Compute KernelSHAP values for deep SPD models and multiple subjects."""
    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating shap_values for {subject_name}...")
        perm = DeepShapleyValuesSPD()
        perm.fit(X, y, n_splits=n_splits, model_config=model_config, model_trained=model, model_type=model_type)
        results_dic[subject_name] = {
        "importance": perm.shap_values_,
        "accuracy": perm.scores_,
    }
    return results_dic

def compute_deep_kernel_shap_eeg_for_subjects(data, n_splits=10, n_perm=10, model_config= None, model = None, model_type = None, n_jobs = -1):
    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating shap_values for {subject_name}...")
        perm = DeepShapleyValuesEEG()
        perm.fit(X, y, n_splits=n_splits, model_config=model_config, model=model, model_type=model_type)
        results_dic[subject_name] = {
        "importance": perm.shap_values_,
        "accuracy": perm.scores_,
    }
    return results_dic


def _with_legacy_result_keys(results):
    """Expose the former result keys for backward-compatible wrappers."""
    return {
        subject: {
            "shap_values": result["importance"],
            "scores": result["accuracy"],
        }
        for subject, result in results.items()
    }


def run_multi_subjects_spd(*args, **kwargs):
    return _with_legacy_result_keys(
        compute_kernel_shap_spd_for_subjects(*args, **kwargs)
    )


def run_multi_subjects_eeg(*args, **kwargs):
    return _with_legacy_result_keys(
        compute_kernel_shap_eeg_for_subjects(*args, **kwargs)
    )


def run_multi_subjects_deep_spd(*args, **kwargs):
    return _with_legacy_result_keys(
        compute_deep_kernel_shap_spd_for_subjects(*args, **kwargs)
    )


def run_multi_subjects_deep_eeg(*args, **kwargs):
    return _with_legacy_result_keys(
        compute_deep_kernel_shap_eeg_for_subjects(*args, **kwargs)
    )
