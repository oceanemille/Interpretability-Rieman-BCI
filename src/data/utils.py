import numpy as np

def split_subjects_by_score(results_dic, threshold=0.75, agg = np.mean):
    high, low = [], []
    for subject_name, res in results_dic.items():
        target = high if agg(res["accuracy"]) >= threshold else low
        target.append(subject_name)
    return high, low

def stack_subjects_by_field(results_dic, subject_names, field):
    return np.array([np.mean(results_dic[subject_name][field],axis=0) for subject_name in subject_names])