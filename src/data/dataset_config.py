from moabb.datasets import BNCI2014_001, Dreyer2023C, Beetl2021_A
import numpy as np

DATASET_CONFIG = {
    "BNCI2014_001": dict(
        dataset=BNCI2014_001(),
        session="0train",
        good_subjects=np.array([1, 3, 8, 9]),
        bad_subjects=np.array([2, 4, 5, 6, 7]),
        sensors =  [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    ]),

    "Dreyer2023C": dict(
        dataset=Dreyer2023C(),
        session="0",
        good_subjects=np.array([83, 85, 87]),
        bad_subjects=np.array([82,84,86]),
        sensors = [
    'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2',
  'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'
]
    ),
    "Beetl2021_A": dict(
        dataset=Beetl2021_A(),
        session="0",
        good_subjects=np.array([1, 3]),
        bad_subjects=np.array([2]),
        sensors = [
                "Fp1",
                "Fz",
                "F3",
                "F7",
                "FT9",
                "FC5",
                "FC1",
                "C3",
                "T7",
                "TP9",
                "CP5",
                "CP1",
                "Pz",
                "P3",
                "P7",
                "O1",
                "Oz",
                "O2",
                "P4",
                "P8",
                "TP10",
                "CP6",
                "CP2",
                "C4",
                "T8",
                "FT10",
                "FC6",
                "FC2",
                "F4",
                "F8",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "F1",
                "F5",
                "FT7",
                "FC3",
                "FCz",
                "C1",
                "C5",
                "TP7",
                "CP3",
                "P1",
                "P5",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "P6",
                "P2",
                "CPz",
                "CP4",
                "TP8",
                "C6",
                "C2",
                "FC4",
                "FT8",
                "F6",
                "F2",
                "AF4",
                "AF8",
            ],
    ),
}
"""

import pickle
import math

OUT_DIR = f'../../Results/Permutation/MDM/BNCI2014_001/With_cue/Modif_covs/Heuristic'


with open(f"{OUT_DIR}/results_per_subject_greedy_variance_diagonal_covs_init.pkl", "rb") as f:
    files_alpha_1_covs_0 = pickle.load(f)

nan_percentage = []
features_greater_than_18_percentage = []
features_less_than_5_percentage = []

for subject in range (len(files_alpha_1_covs_0)):
    number_of_features = []
    number_of_nan = 0
    print("Subject : ", subject)
    for i in range (10):
        for dic in files_alpha_1_covs_0[subject][i]:
            print(dic['selected_features'])
            if math.isnan(dic['n_features_needed']):
                number_of_nan += 1
            else :
                number_of_features.append(dic['n_features_needed'])

    number_of_features = np.array(number_of_features)
    nan_percentage.append(number_of_nan / (10 * len(files_alpha_1_covs_0[subject][0])) * 100)
    features_greater_than_18_percentage.append(np.sum(number_of_features >= 18) / (10 * len(files_alpha_1_covs_0[subject][0])) * 100)
    features_less_than_5_percentage.append(np.sum(number_of_features <= 5) / (10 * len(files_alpha_1_covs_0[subject][0])) * 100)

    print("Number of nan : ", number_of_nan)
    print("Mean", np.mean(number_of_features))
    print("Median", np.median(number_of_features))
    print("Number of features >= 18", np.sum(number_of_features >= 18))
    print("Number of features <= 5", np.sum(number_of_features <=5))
    print("Percentage of NaN values: ", nan_percentage[-1])
    print("Percentage of feature counts >= 18: ", features_greater_than_18_percentage[-1])
    print("Percentage of feature counts <= 5: ", features_less_than_5_percentage[-1])

print("Percentage of NaN values: ", np.mean(nan_percentage))
print("Percentage of feature counts >= 18: ", np.mean(features_greater_than_18_percentage))
print("Percentage of feature counts <= 5: ", np.mean(features_less_than_5_percentage))


with open(f"{OUT_DIR}/results_per_subject_opposite_variance_opposite.pkl", "rb") as f:
    files_alpha_1 = pickle.load(f)

with open(f"{OUT_DIR}/results_per_subject_variance_unchanged.pkl", "rb") as f:
    files_alpha_0 = pickle.load(f)

print(files_alpha_1[0][0], '\n')
print(files_alpha_0[0][0])


with open(f"{OUT_DIR}/results_per_subject_covs_init_modif_var_code_1.pkl", "rb") as f:
    files = pickle.load(f)

nan_percentage = []
features_greater_than_18_percentage = []
features_less_than_5_percentage = []

for subject in range (len(files)):
    number_of_features = []
    number_of_nan = 0
    print("Subject : ", subject)
    for i in range (10):
        for dic in files[subject][i]:
            print(dic['selected_features'])
            if math.isnan(dic['n_features_needed']):
                number_of_nan += 1
            else :
                number_of_features.append(dic['n_features_needed'])

    number_of_features = np.array(number_of_features)
    nan_percentage.append(number_of_nan / (10 * len(files[subject][0])) * 100)
    features_greater_than_18_percentage.append(np.sum(number_of_features >= 18) / (10 * len(files[subject][0])) * 100)
    features_less_than_5_percentage.append(np.sum(number_of_features <= 5) / (10 * len(files[subject][0])) * 100)

    print("Number of nan : ", number_of_nan)
    print("Mean", np.mean(number_of_features))
    print("Median", np.median(number_of_features))
    print("Number of features >= 18", np.sum(number_of_features >= 18))
    print("Number of features <= 5", np.sum(number_of_features <=5))
    print("Percentage of NaN values: ", nan_percentage[-1])
    print("Percentage of feature counts >= 18: ", features_greater_than_18_percentage[-1])
    print("Percentage of feature counts <= 5: ", features_less_than_5_percentage[-1])

print("Percentage of NaN values: ", np.mean(nan_percentage))
print("Percentage of feature counts >= 18: ", np.mean(features_greater_than_18_percentage))
print("Percentage of feature counts <= 5: ", np.mean(features_less_than_5_percentage))


with open(f"{OUT_DIR}/results_per_subject_variance_covariance.pkl", "rb") as f:
    files_comp = pickle.load(f)

from collections import defaultdict



for subject in range (len(files_comp)):
    counter = defaultdict(int)
    number_of_features = []
    number_of_nan = 0
    print("Subject : ", subject)
    for i in range (10):
        for dic in files_comp[subject][i]:
            flipped = dic['flipped']
            signature = (
                flipped['original'],
                flipped['covariance_only'],
                flipped['variance_only'],
                flipped['combined']
            )
            counter[signature] += 1
    print("Counter : ", counter)

with open(f"{OUT_DIR}/results_per_subject_covs_init_variance_modif_alpha_opti.pkl", "rb") as f:
    files_alpha_opti = pickle.load(f)


for subject in range (len(files_alpha_opti)):
    number_of_features = []
    number_of_nan = 0
    print("Subject : ", subject)
    for i in range (10):
        for dic in files_alpha_opti[subject][i]:
            print(dic['alpha_min'])

"""
