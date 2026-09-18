import numpy as np


def build_results_dict(subject_names, importance, accuracy):
    """Build the common multi-subject result structure from saved arrays."""
    subject_names = list(subject_names)
    importance = np.asarray(importance)
    accuracy = np.asarray(accuracy)
    if importance.ndim == 0 or accuracy.ndim == 0:
        raise ValueError("importance and accuracy must have a subject axis.")
    if len(subject_names) != importance.shape[0] or len(subject_names) != accuracy.shape[0]:
        raise ValueError(
            "subject_names, importance and accuracy must contain the same "
            "number of subjects."
        )

    return {
        subject: {
            "importance": importance[index],
            "accuracy": np.atleast_1d(accuracy[index]),
        }
        for index, subject in enumerate(subject_names)
    }

def split_subjects_by_score(results_dic, threshold=0.75, agg = np.mean):
    high, low = [], []
    for subject_name, res in results_dic.items():
        target = high if agg(res["accuracy"]) >= threshold else low
        target.append(subject_name)
    return high, low

def stack_subjects_by_field(results_dic, subject_names, field, axis=0):
    """Aggregate a result field and stack the selected subjects."""
    return np.array([
        np.mean(results_dic[subject_name][field], axis=axis)
        for subject_name in subject_names
    ])
