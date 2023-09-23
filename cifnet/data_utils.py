"""
This code includes the functions to process the data ant prepare it for the metric learning.
"""
import pandas as pd
import torch
import numpy as np
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transforms import AffineTransform

from cifnet.scm_synthetic import SCM_LIN, SCM_NLM, SCM_Loan, SCM_IMF
from cifnet.utils import get_data_file, scms_save_dir
from cifnet.scm_datasets import Learned_Adult_SCM, Learned_COMPAS_SCM
from cifnet.dissimilarities import l_p
pd.options.mode.chained_assignment = None


def get_scm(data_name):
    """
    This function returns the SCM for the given dataset name.
    Inputs:
        data_name: str, name of the dataset includes compas, adult, lin, nlm, imf, loan
    Outputs:
        scm: SCM object
    """
    learned_scms = {'adult': Learned_Adult_SCM, 'compas': Learned_COMPAS_SCM}

    if data_name in learned_scms.keys():
        # print('Fitting SCM for %s...' % data_name)

        np.random.seed(0)
        torch.manual_seed(0)

        X, _, _ = process_data(data_name)
        scm = learned_scms[data_name](linear=False)
        scm.fit_eqs(X.to_numpy(), save=scms_save_dir + data_name)
    elif data_name == 'lin':
        scm = SCM_LIN()
    elif data_name == 'nlm':
        scm = SCM_NLM()
    elif data_name == 'imf':
        scm = SCM_IMF()
    elif data_name == 'loan':
        scm = SCM_Loan()
    else:
        raise AssertionError
    return scm


def learning_data(data_name, data_type, num_points=1000, margin=1.0, Q_metric=l_p, sensitive=None):
    """
    This function returns the learning data for the given dataset name.
    :param data_name: data name includes compas, adult, lin, nlm, imf, loan
    :param data_type: contrastive, triplet
    :param num_points: number of points to generate
    :param margin: margin for triplet loss or diameter for distance loss
    :param Q_metric: semi-latent metric
    :param sensitive: the list of sensitive attributes of the dataset
    :return: The learning data for the given dataset name.
    """

    scm = get_scm(data_name)
    if data_type == "contrastive":
        data_dict = scm.generate_contrastive(num_points, margin, Q_metric)
    elif data_type == "triplet":
        data_dict = scm.generate_triplet(num_points, margin, Q_metric)
    else:
        raise AssertionError

    return data_dict


def dict_to_train_test(data_dict, ratio=0.8, tensor=True, device='cpu'):
    """
    This function converts the data dictionary that includes different data to train and test sets.
    :param data_dict: dictionary that includes np.array (N, D) data for each key
    :param ratio: ratio of the train data
    :param tensor: if True, convert the data to torch.Tensor
    :param device: device to put the data
    :return: train_dict: dictionary that includes np.array (M, D) train data for each key
             test_dict: dictionary that includes np.array (N-M, D) test data for each key
    """
    # Shuffle indices
    _, X = next(iter(data_dict.items()))
    N_data = X.shape[0]
    indices = np.random.choice(np.arange(N_data), size=N_data, replace=False)
    N_train = int(N_data * ratio)
    train_dict = {}
    test_dict = {}
    for key in data_dict.keys():
        if tensor:
            train_dict[key] = torch.Tensor(data_dict[key][indices[:N_train]]).to(torch.device(device))
            test_dict[key] = torch.Tensor(data_dict[key][indices[N_train:]]).to(torch.device(device))
        else:
            train_dict[key] = data_dict[key][indices[:N_train]]
            test_dict[key] = data_dict[key][indices[N_train:]]

    return train_dict, test_dict


"""
Below functions adopted from https://github.com/RicardoDominguez/AdversariallyRobustRecourse/tree/main/src repository
and modified to fit our needs.
"""

def process_data(data, n_sample=1000):
    if data == "compas":
        return process_compas_causal_data()
    elif data == "adult":
        return process_causal_adult()
    else:
        raise AssertionError
def process_compas_causal_data():
    data_file = get_data_file("compas-scores-two-years")
    compas_df = pd.read_csv(data_file, index_col=0)

    # Standard way to process the data, as done in the ProPublica notebook
    compas_df = compas_df.loc[(compas_df['days_b_screening_arrest'] <= 30) &
                              (compas_df['days_b_screening_arrest'] >= -30) &
                              (compas_df['is_recid'] != -1) &
                              (compas_df['c_charge_degree'] != "O") &
                              (compas_df['score_text'] != "NA")]
    compas_df['age'] = (pd.to_datetime(compas_df['c_jail_in']) - pd.to_datetime(compas_df['dob'])).dt.days / 365

    # We use the variables in the causal graph of Nabi & Shpitser, 2018
    X = compas_df[['age', 'race', 'sex', 'priors_count']]
    X['isMale'] = X.apply(lambda row: 1 if 'Male' in row['sex'] else 0, axis=1)
    X['isCaucasian'] = X.apply(lambda row: 1 if 'Caucasian' in row['race'] else 0, axis=1)
    X = X.drop(['sex', 'race'], axis=1)

    # Swap order of features to simplify learning the SCM
    X = X[['age', 'isMale', 'isCaucasian', 'priors_count']]

    # Favourable outcome is no recidivism
    y = compas_df.apply(lambda row: 1.0 if row['two_year_recid'] == 0 else 0.0, axis=1)

    columns = X.columns
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]

    # Only the number of prior counts is actionable
    compas_actionable_features = ["priors_count"]
    actionable_ids = [idx for idx, col in enumerate(X.columns) if col in compas_actionable_features]

    # Number of priors cannot go below 0
    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'priors_count')[0]] = np.array([0, 10e10])

    # Standardize continuous features
    compas_categorical_names = ['isMale', 'isCaucasian']
    for col_idx, col in enumerate(X.columns):
        if col not in compas_categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    # Get the indices for increasing and decreasing features
    compas_increasing_actionable_features = []
    compas_decreasing_actionable_features = ["priors_count"]
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in compas_increasing_actionable_features]
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in compas_decreasing_actionable_features]
    sensitive = [1, 2]  # ['isMale', 'isCaucasian']

    constraints = {'sensitive': sensitive, 'actionable': actionable_ids, 'increasing': increasing_ids,
                   'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints


def process_causal_adult():
    # TODO: replace old data with ASCIncome dataset
    data_file = get_data_file("adult")
    adult_df = pd.read_csv(data_file).reset_index(drop=True)
    adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'label']  # proper name of each of the features
    adult_df = adult_df.dropna()

    #  We use the variables in the causal graph of Nabi & Shpitser, 2018
    adult_df = adult_df.drop(['workclass', 'fnlwgt', 'education', 'occupation', 'relationship', 'race', 'capital-gain',
                              'capital-loss'], axis=1)
    adult_df['native-country-United-States'] = adult_df.apply(
        lambda row: 1 if 'United-States' in row['native-country'] else 0, axis=1)
    adult_df['marital-status-Married'] = adult_df.apply(lambda row: 1 if 'Married' in row['marital-status'] else 0,
                                                        axis=1)
    adult_df['isMale'] = adult_df.apply(lambda row: 1 if 'Male' in row['sex'] else 0, axis=1)
    adult_df = adult_df.drop(['native-country', 'marital-status', 'sex'], axis=1)
    X = adult_df.drop('label', axis=1)

    # Target is whether the individual has a yearly income greater than 50k
    y = adult_df['label'].replace(' <=50K', 0.0)
    y = y.replace(' >50K', 1.0)

    # Re-arange to follow the causal graph
    columns = ['isMale', 'age', 'native-country-United-States', 'marital-status-Married', 'education-num',
               'hours-per-week']
    X = X[columns]

    adult_actionable_features = ["education-num", "hours-per-week"]
    actionable_ids = [idx for idx, col in enumerate(X.columns) if col in adult_actionable_features]

    feature_limits = np.array([[-1, 1]]).repeat(X.shape[1], axis=0) * 1e10
    feature_limits[np.where(X.columns == 'education-num')[0]] = np.array([1, 16])
    feature_limits[np.where(X.columns == 'hours-per-week')[0]] = np.array([0, 100])

    # Standardize continuous features#
    means = [0 for i in range(X.shape[-1])]
    std = [1 for i in range(X.shape[-1])]
    adult_categorical_names = ['isMale', 'native-country-United-States', 'marital-status-Married']
    for col_idx, col in enumerate(X.columns):
        if col not in adult_categorical_names:
            means[col_idx] = X[col].mean(axis=0)
            std[col_idx] = X[col].std(axis=0)
            X[col] = (X[col] - X[col].mean(axis=0)) / X[col].std(axis=0)
            feature_limits[col_idx] = (feature_limits[col_idx] - means[col_idx]) / std[col_idx]

    adult_increasing_actionable_features = ["education-num"]
    adult_decreasing_actionable_features = []
    increasing_ids = [idx for idx, col in enumerate(X.columns) if col in adult_increasing_actionable_features]
    decreasing_ids = [idx for idx, col in enumerate(X.columns) if col in adult_decreasing_actionable_features]
    sensitive = [0]  # ['isMale']
    constraints = {'sensitive': sensitive, 'actionable': actionable_ids, 'increasing': increasing_ids,
                   'decreasing': decreasing_ids, 'limits': feature_limits}

    return X, y, constraints


class StandardLogisticDistribution:

    def __init__(self, input_dim, device='cpu'):
        self.m = TransformedDistribution(
            Uniform(torch.zeros(input_dim, device=device),
                    torch.ones(input_dim, device=device)),
            [SigmoidTransform().inv, AffineTransform(torch.zeros(input_dim, device=device),
                                                     torch.ones(input_dim, device=device))]
        )

    def log_pdf(self, z):
        return self.m.log_prob(z).sum(dim=1)

    def sample(self):
        return self.m.sample()