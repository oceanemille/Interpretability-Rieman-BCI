from .dataset_config import DATASET_CONFIG
from moabb.paradigms import FilterBankMotorImagery
from pyriemann.estimation import Covariances

def load_from_moabb(dataset, covs = False, paradigm=None, sensors=None):
    """
    Parameters
    ----------
    dataset : str ou objet MOABB Dataset
        Si str : doit être une clé de DATASET_CONFIG (datasets pré-configurés
        avec leurs capteurs). Si objet Dataset MOABB directement : doit être
        fourni avec `sensors`.
    sensors : list, optionnel
        Requis si `dataset` est un objet MOABB (pas dans DATASET_CONFIG).
    covs : bool, optionnel
        Si covs est positif, la fonction renvoie les matrices de covariances de chaque trials par sujet
        Sinon, elle renvoie les données brutes par sujet
    """
    if isinstance(dataset, str):
        if dataset not in DATASET_CONFIG:
            raise ValueError(
                f"'{dataset}' n'est pas dans DATASET_CONFIG. "
                f"Datasets disponibles : {list(DATASET_CONFIG.keys())}. "
                f"Pour utiliser un autre dataset MOABB, passez directement "
                f"l'objet dataset et la liste des capteurs via `sensors=`."
            )
        cfg = DATASET_CONFIG[dataset]
        dataset_obj = cfg["dataset"]
        sensors = cfg["sensors"]
    else:
        dataset_obj = dataset
        if sensors is None:
            raise ValueError(
                "Vous devez fournir `sensors` quand vous passez un objet "
                "Dataset MOABB directement (pas dans DATASET_CONFIG)."
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