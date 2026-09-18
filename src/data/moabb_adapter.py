from .dataset_config import DATASET_CONFIG
from moabb.paradigms import FilterBankMotorImagery
from pyriemann.estimation import Covariances

def load_from_moabb(dataset, covs = False, paradigm=None, sensors=None):
    """Load one or more subjects from a MOABB dataset.

    Parameters
    ----------
    dataset : str or MOABB Dataset
        A string must be a key from ``DATASET_CONFIG``. A MOABB dataset object
        can be supplied directly together with ``sensors``.
    sensors : list, optional
        Required when ``dataset`` is not configured in ``DATASET_CONFIG``.
    covs : bool, optional
        If ``True``, return one covariance matrix per trial. Otherwise, return
        raw EEG trials.
    """
    if isinstance(dataset, str):
        if dataset not in DATASET_CONFIG:
            raise ValueError(
                f"'{dataset}' is not in DATASET_CONFIG. "
                f"Available datasets: {list(DATASET_CONFIG.keys())}. "
                "To use another MOABB dataset, pass the dataset object "
                "directly and provide its channel names through `sensors=`."
            )
        cfg = DATASET_CONFIG[dataset]
        dataset_obj = cfg["dataset"]
        sensors = cfg["sensors"]
    else:
        dataset_obj = dataset
        if sensors is None:
            raise ValueError(
                "Provide `sensors` when passing a MOABB dataset object that "
                "is not configured in DATASET_CONFIG."
            )

    paradigm = paradigm or FilterBankMotorImagery(filters=[[7, 35]], events={"left_hand": 1, "right_hand": 2})

    data = {}
    for subject in dataset_obj.subject_list:
        X, y, metadata = paradigm.get_data(dataset=dataset_obj, subjects=[subject])
        if covs :
            covs_transform = Covariances()
            X = covs_transform.fit_transform(X)
        data[f"Subject {subject}"] = (X, y)

    return data, sensors
