from .ExactShapleyValues import fit

def run_multi_subjects_spd(
    data,
    clf=None,
    n_splits=10,
    *,
    pretrained=False,
    input_is_covariance=None,
    baseline=None,
    model_config=None,
    batch_size=20_000,
    verbose=True,
):

    results_dic = {}
    for subject_name, (X, y) in data.items():
        print(f"Calculating importance for {subject_name}...")
        shapley_values, scores = fit(X,y,clf=clf, n_splits = n_splits, pretrained = pretrained, 
        input_is_covariance = input_is_covariance, baseline=baseline, model_config=model_config,
        batch_size=batch_size, verbose = verbose)
        results_dic[subject_name] = {
        "importance": shapley_values,  
        "accuracy": scores,
    }

    return results_dic  
            


