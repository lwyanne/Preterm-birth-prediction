import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import logging
import re
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.metrics import log_loss
import numpy as np

from pathlib import Path
import os
from fastai.callbacks import *
from fastai.tabular import *
pj = os.path.join
path='/home/shuying/predictBirth/code'
data_path = '/home/shuying/predictBirth/'
results_path = '/home/shuying/predictBirth/code/results'
model_path='/home/shuying/predictBirth/code/models'
cat_nul=['mbstate_rec',
                 # 'mbrace',
                 # 'mhisp_r',
         'mracehisp',

         'mar_p',
                 'dmar',
                 'meduc',
                 'fracehisp',
                 'feduc',
                 'wic',
                 'rf_pdiab',
                 'rf_phype',
                 # 'rf_ppterm',
                 'rf_inftr',
                 'rf_fedrg',
                 'rf_artec',
                 'ip_gon',
                 'ip_syph',
                 'ip_chlam',
                 'ip_hepatb',
                 'ip_hepatc',
                 'dplural',
                 'sex'
                 ]
con_nul= ['mager',
        # 'fagecomb',
        'precare',
        'cig_0',
        'cig_1',
        'm_ht_in',
        'bmi',
        # 'rf_cesarn',
          ]
cat_mul=['mbstate_rec',
                 # 'mbrace',
                 # 'mhisp_r',
                 'mar_p',
         'mracehisp',

         'dmar',
                 'meduc',
                 'fracehisp',
                 'feduc',
                 'wic',
                 'rf_pdiab',
                 # 'rf_gdiab',
                 'rf_phype',
                 'rf_ppterm',
                 'rf_inftr',
                 'rf_fedrg',
                 'rf_artec',
                 'ip_gon',
                 'ip_syph',
                 'ip_chlam',
                 'ip_hepatb',
                 'ip_hepatc',
                 'dplural',
                 'sex'

                 ]
con_mul= [
        'mager',
        # 'fagecomb',
        'priorlive',
        'priordead',
        'priorterm',
        'illb_r',
        'ilop_r',
        'ilp_r',
        'precare',
        'cig_0',
        'cig_1',
        #  'cig_2',
        # 'cig_3',
        'm_ht_in',
        'bmi',
        'rf_cesarn',
    ]
def get_ci(m):
    return float(re.match(r'.*(0\.[0-9]*).*(0\.[0-9]*).*', m).group(1)), float(
        re.match(r'.*(0\.[0-9]*).*(0\.[0-9]*).*', m).group(2))

def get_variables_nul():
    cat_names = ['mbstate_rec',
                 # 'mbrace',
                 # 'mhisp_r',
                 'mar_p',
                 'mracehisp',

                 'dmar',
                 'meduc',
                 'fracehisp',
                 'feduc',
                 'wic',
                 'rf_pdiab',
                 'rf_phype',
                 # 'rf_ppterm',
                 'rf_inftr',
                 'rf_fedrg',
                 'rf_artec',
                 'ip_gon',
                 'ip_syph',
                 'ip_chlam',
                 'ip_hepatb',
                 'ip_hepatc',
                 'dplural',
                 'sex'
                 ]
    cont_names = [
        'mager',
        # 'fagecomb',
        'precare',
        'cig_0',
        'cig_1',
        'm_ht_in',
        'bmi',
        # 'rf_cesarn',
    ]
    dep_var = 'cat3'
    variables = cont_names + cat_names
    embdic = {'mbstate_rec': 3,
              # 'mbrace': 3,
              # 'mhisp_r': 4,
              'mar_p': 3,
              'dmar': 2,
              'meduc': 4,
              'fracehisp': 4,
              'mracehisp':4,
             'feduc': 4,
              'wic': 2,
              'rf_pdiab': 2,
              'rf_phype': 2,
              'rf_ppterm': 2,
              'rf_inftr': 2,
              'rf_fedrg': 2,
              'rf_artec': 2,
              'ip_gon': 2,
              'ip_syph': 2,
              'ip_chlam': 2,
              'ip_hepatb': 2,
              'ip_hepatc': 2,
              'dplural': 2,
              'sex': 2}
    return cat_names,cont_names,variables,dep_var,embdic


def get_variables_mul():
    cat_names = ['mbstate_rec',
                 # 'mbrace',
                 # 'mhisp_r',
                 'mracehisp',

                 'mar_p',
                 'dmar',
                 'meduc',
                 'fracehisp',
                 'feduc',
                 'wic',
                 'rf_pdiab',
                 # 'rf_gdiab',
                 'rf_phype',
                 'rf_ppterm',
                 'rf_inftr',
                 'rf_fedrg',
                 'rf_artec',
                 'ip_gon',
                 'ip_syph',
                 'ip_chlam',
                 'ip_hepatb',
                 'ip_hepatc',
                 'dplural',
                 'sex'

                 ]
    cont_names = [
        'mager',
        # 'fagecomb',
        'priorlive',
        'priordead',
        'priorterm',
        'illb_r',
        'ilop_r',
        'ilp_r',
        'precare',
        'cig_0',
        'cig_1',
        #  'cig_2',
        # 'cig_3',
        'm_ht_in',
        'bmi',
        'rf_cesarn',
    ]
    dep_var = 'cat3'
    variables = cat_names + cont_names
    embdic = {'mbstate_rec': 3,
              # 'mbrace': 3,
              # 'mrace31': 15,
              'mracehisp':4,

             # 'mhisp_r': 4,
              'mar_p': 3,
              'dmar': 2,
              'meduc': 4,
              'fhisp_r': 4,
              'fracehisp': 4,
              'feduc': 4,
              'wic': 2,
              'rf_pdiab': 2,
              'rf_gdiab': 2,
              'rf_phype': 2,
              'rf_ppterm': 2,
              'rf_inftr': 2,
              'rf_fedrg': 2,
              'rf_artec': 2,
              'ip_gon': 2,
              'ip_syph': 2,
              'ip_chlam': 2,
              'ip_hepatb': 2,
              'ip_hepatc': 2,
              'dplural': 2,
              'sex': 2}
    return cat_names,cont_names,variables,dep_var,embdic

def one_hot(df,cat_names):
    logger = logging.getLogger('logger')
    df = pd.get_dummies(df,columns=cat_names,dummy_na=True)
    logger.info('after one-hot encoding, data frame looks like/n%s'%df.head())
    return df


def fill_missing(df,log=False):
    logger=logging.getLogger('logger')
    cat = ['mbstate_rec',
           # 'mbrace',
           # 'mhisp_r',
           'mar_p',
           'dmar',
           'meduc',
           'fracehisp',
           'mracehisp',

           'feduc',
           'wic',
           'rf_pdiab',
           'rf_phype',
           'rf_ppterm',
           'rf_inftr',
           'rf_fedrg',
           'rf_artec',
           'ip_gon',
           'ip_syph',
           'ip_chlam',
           'ip_hepatb',
           'ip_hepatc',
           'dplural',
           'sex'
           ]
    con = ['mager',
           # 'fagecomb',
           'priorlive',
           'priordead',
           'priorterm',
           'illb_r',
           'ilop_r',
           'ilp_r',
           'precare',
           'cig_0',
           'cig_1',
           'm_ht_in',
           'bmi',
           'rf_cesarn']
    for col in df.columns:
        if col in con and 'cat' not in col:
            m=df[col].mean()
            df[col]=df[col].replace(np.nan, m)
            if log:
                logger.info('Missing values in %s is imputed as %s'%(col,m) )
        # elif col in cat:
        #     m=df[col].mode().iloc[0]
        #     df[col]=df[col].replace(np.nan, m)
        #     if log:
        #         logger.info('Missing values in %s is imputed as %s'%(col,m) )
        elif col in cat and 'cat' not in col:
            df[col] = df[col].replace(np.nan, 'NAN')
            if log:
                logger.info('Missing values in %s is imputed as %s'%(col,m) )
    return df



def compute_midrank(x):
    """
    Computes midranks.
    Parameters
    ----------
        x : np.array
            x - a 1D numpy array
    Returns
    -------
       T2 : np.array
           Array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1

    return T2


def compute_midrank_weight(x, sample_weight):
    """
    Computes midranks.
    Parameters
    ----------
        x : np.array
        sample_weigh : int
            x - a 1D numpy array
    Returns
    -------
        T2 : np.array
            array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(
            predictions_sorted_transposed,
            label_1_count)
    else:
        return fastDeLong_weights(
            predictions_sorted_transposed,
            label_1_count,
            sample_weight)


def fastDeLong_weights(pred_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Reference
    ----------
    @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    Parameters
    ----------
       predictions_sorted_transposed : np.array
           a 2D numpy.array[n_classifiers, n_examples] sorted such as the
           examples with label "1" are first
    Returns
    -------
        aucs : float
        delongcov : float
            (AUC value, DeLong covariance)
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = pred_sorted_transposed.shape[1] - m
    positive_examples = pred_sorted_transposed[:, :m]
    negative_examples = pred_sorted_transposed[:, m:]
    k = pred_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(
            positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(
            negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(
            pred_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(
        sample_weight[:m, np.newaxis],
        sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (
                   sample_weight[:m] * (tz[:, :m] - tx)
           ).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Reference:
        @article{sun2014fast,
             title={
                 Fast Implementation of DeLong's Algorithm for
                 Comparing the Areas Under Correlated Receiver Oerating
                 Characteristic Curves},
             author={Xu Sun and Weichao Xu},
             journal={IEEE Signal Processing Letters},
             volume={21},
             number={11},
             pages={1389--1393},
             year={2014},
             publisher={IEEE}
         }
    Parameters
    ----------
        predictions_sorted_transposed : ?
        label_1_count : ?
            predictions_sorted_transposed: a 2D
            ``numpy.array[n_classifiers, n_examples]``
            sorted such as the examples with label "1" are first
    Returns
    -------
       (AUC value, DeLong covariance)
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """
    Computes log(10) of p-values.
    Parameters
    ----------
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns
    -------
       log10(pvalue)
    """
    l_aux = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l_aux, sigma), l_aux.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Parameters
    ----------
        ground_truth: np.array
            of 0 and 1
        predictions: np.array
            of floats of the probability of being class 1
    """
    ground_truth_stats = compute_ground_truth_statistics(
        ground_truth,
        sample_weight)
    order, label_1_count, ordered_sample_weight = ground_truth_stats

    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(
        predictions_sorted_transposed,
        label_1_count,
        ordered_sample_weight)

    assert_msg = "There is a bug in the code, please forward this to the devs"
    assert len(aucs) == 1, assert_msg
    return aucs[0], delongcov


def delong_roc_test(ground_truth, pred_one, pred_two, sample_weight=None):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Parameters
    ----------
       ground_truth: np.array
           of 0 and 1
       predictions_one: np.array
           predictions of the first model,
           np.array of floats of the probability of being class 1
       predictions_two: np.array
           predictions of the second model, np.array of floats of the
           probability of being class 1
    """
    order, label_1_count, _ = compute_ground_truth_statistics(
        ground_truth,
        sample_weight)

    predictions_sorted_transposed = np.vstack(
        (pred_one, pred_two))[:, order]

    aucs, delongcov = fastDeLong(
        predictions_sorted_transposed,
        label_1_count,
        sample_weight)

    # print(aucs, delongcov)
    return calc_pvalue(aucs, delongcov)


def auc_ci_Delong(y_true, y_scores, alpha=.95):
    """AUC confidence interval via DeLong.
    Computes de ROC-AUC with its confidence interval via delong_roc_variance
    References
    -----------
        See this `Stack Overflow Question
        <https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals/53180614#53180614/>`_
        for further details
    Examples
    --------
    ::
        y_scores = np.array(
            [0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
        y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])
        auc, auc_var, auc_ci = auc_ci_Delong(y_true, y_scores, alpha=.95)
        np.sqrt(auc_var) * 2
        max(auc_ci) - min(auc_ci)
        print('AUC: %s' % auc, 'AUC variance: %s' % auc_var)
        print('AUC Conf. Interval: (%s, %s)' % tuple(auc_ci))
        Out:
            AUC: 0.8 AUC variance: 0.028749999999999998
            AUC Conf. Interval: (0.4676719375452081, 1.0)
    Parameters
    ----------
    y_true : list
        Ground-truth of the binary labels (allows labels between 0 and 1).
    y_scores : list
        Predicted scores.
    alpha : float
        Default 0.95
    Returns
    -------
        auc : float
            AUC
        auc_var : float
            AUC Variance
        auc_ci : tuple
            AUC Confidence Interval given alpha
    """

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Get AUC and AUC variance
    auc, auc_var = delong_roc_variance(
        y_true,
        y_scores)

    auc_std = np.sqrt(auc_var)

    # Confidence Interval
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    lower_upper_ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    lower_upper_ci[lower_upper_ci > 1] = 1

    return auc, auc_var, lower_upper_ci


def setup_logs(run_name,type='logger'):
    """

    :param save_dir:  the directory to set up logs
    :param type:  'model' for saving logs in 'logs/cpc'; 'imp' for saving logs in 'logs/imp'
    :param run_name:
    :return:logger
    """
    # initialize logger
    logger = logging.getLogger(type)
    logger.setLevel(logging.INFO)

    # create the logging file handler
    log_file = os.path.join('/home/shuying/predictBirth/code/results', run_name + ".log")
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def read_train_val_nul(fillna='manual', split_x_y=False):
    df = pd.read_csv(pj(data_path, 'nul_1417_even.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if fillna=='manual':
        df=one_hot(df,cat_nul)

        df = fill_missing(df)

    if split_x_y:
        return df.iloc[:, :].drop('cat3',axis=1), df.iloc[:, :]['cat3']
    else:
        df = fill_missing(df)

        return df.iloc[:, :]

def read_train_nul(fillna='manual', split_x_y=False):
    df = pd.read_csv(pj(data_path, 'nul_1417_even.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if fillna=='manual':
        df=one_hot(df,cat_nul)
        df = fill_missing(df,True)

    if split_x_y:
        return df.iloc[:-int(0.5e6), :].drop('cat3',axis=1), df.iloc[:-int(0.5e6), :]['cat3']
    else:
        return df.iloc[:-int(0.5e6), :]


def read_val_nul(fillna='manual', split_x_y=False):
    df = pd.read_csv(pj(data_path, 'nul_1417_even.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if fillna=='manual':
        df=one_hot(df,cat_nul)
        df = fill_missing(df)

    if split_x_y:
        return df.iloc[-int(0.5e6):, :].drop('cat3',axis=1), df.iloc[-int(0.5e6):, :]['cat3']
    else:
        return df.iloc[-int(0.5e6):, :]


def read_test_nul(fillna='manual', split_x_y=False):
    df = pd.read_csv(pj(data_path, 'nulli_2018.csv'))
    df_tmp=pd.read_csv(pj(data_path, 'nul_1417_even.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df_tmp.drop(df_tmp.columns[df_tmp.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    l=len(df)
    if fillna=='manual':
        df=one_hot(pd.concat((df,df_tmp),axis=0),cat_nul)
        df = fill_missing(df)

    if split_x_y:
        return df.iloc[:l, :].drop('cat3',axis=1), df.iloc[:l, :]['cat3']
    else:
        df = fill_missing(df)

        return df.iloc[:, :]


def read_train_val_mul(fillna='manual', split_x_y=False):
    df = pd.read_csv(pj(data_path, 'mul_1417_even.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if fillna=='manual':
        df=one_hot(df,cat_mul)

        df = fill_missing(df)

    if split_x_y:
        return df.iloc[:, :].drop('cat3',axis=1), df.iloc[:, :]['cat3']
    else:
        df = fill_missing(df)

        return df.iloc[:, :]


def read_train_mul(fillna='manual', split_x_y=False):
    df = pd.read_csv(pj(data_path, 'mul_1417_even.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if fillna:
        df=one_hot(df,cat_mul)

        df = fill_missing(df, True)

    if split_x_y:
        return df.iloc[:int(-5e5), :].drop('cat3',axis=1), df.iloc[:int(-5e5), :]['cat3']
    else:

        return df.iloc[:int(-5e5), :]


def read_val_mul(fillna='manual', split_x_y=False):
    df = pd.read_csv(pj(data_path, 'mul_1417_even.csv'))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if fillna=='manual':
        df=one_hot(df,cat_mul)

        df = fill_missing(df)

    if split_x_y:
        return df.iloc[-int(5e5):, :].drop('cat3',axis=1), df.iloc[-int(5e5):, :]['cat3']
    else:
        return df.iloc[-int(5e5):, :]


def read_test_mul(fillna='manual', split_x_y=False):
    df = pd.read_csv(pj(data_path, 'multi_2018.csv'))
    df_tmp = pd.read_csv(pj(data_path, 'mul_1417_even.csv'))
    l=len(df)

    df_tmp.drop(df_tmp.columns[df_tmp.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    if fillna=='manual':
        df=one_hot(pd.concat((df,df_tmp),axis=0),cat_mul)

        df = fill_missing(df)
    else:
        df = fill_missing(df)

    if split_x_y:
        return df.iloc[:l, :].drop('cat3',axis=1), df.iloc[:l, :]['cat3']
    else:
        return df.iloc[:l, :]


from sklearn.metrics import roc_auc_score

def auroc_score(input, target):
    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()
    return roc_auc_score(target, input)

class AUROC(Callback):
    """
    This is for output AUROC as a metric in fastai training process.
    This has a small but acceptable issue. #TODO
    """
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn, **kwargs):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['AUROC'])

    def on_epoch_begin(self, **kwargs):
        self.output, self.target = [], []

    def on_batch_end(self, last_target, last_output, train, **kwargs):
        if not train:
            try:
                self.output.append(last_output)
            except AttributeError:
                self.output = []
            try:
                self.target.append(last_target)
            except AttributeError:
                self.target = []

    def on_epoch_end(self, last_metrics, **kwargs):
        if len(self.output) > 0:
            output = torch.cat(self.output).cpu()
            target = torch.cat(self.target).cpu()
            preds = F.softmax(output, dim=1)
            metric = auroc_score(preds,target)
            return add_metrics(last_metrics, [metric])

def get_val_results(learner):
    y = learner.get_preds(ds_type=DatasetType.Valid)
    y2 = y[1]
    y1 = y[0]
    Y2 = 1 - to_np(y2)
    Y1 = to_np(y1[:, 0])
    auc, auc_var, ci = auc_ci_Delong(y_true=Y2, y_scores=Y1)
    return auc, ci


def eval_infection(learner,bestLearner,cat_dic,cont_dic,auc_val):
    logger=logging.getLogger('importance')
    learner.load(bestLearner)
    for item in [(k, v.shape) for k, v in learner.model.state_dict().items()][
                cat_dic['ip_gon']:cat_dic['ip_hepatc'] + 1]:
        new_state_dict = OrderedDict({item[0]: torch.zeros(item[1], dtype=torch.int32, device='cuda:0')})
        learner.model.load_state_dict(new_state_dict, strict=False)
    learner.model.load_state_dict(new_state_dict, strict=False)
    auc_tmp, _ = get_val_results(learner)
    per = (auc_tmp - auc_val) / auc_val * 100
    logger.info('Infection Present: %.3f (%.2f)' % (auc_tmp, per))


def eval_demographic(learner,bestLearner,cat_dic,cont_dic,auc_val):
    logger=logging.getLogger('importance')
    learner.load(bestLearner)
    for item in [(k, v.shape) for k, v in learner.model.state_dict().items()][
                cat_dic['mbstate_rec']:cat_dic['feduc'] + 1]:
        new_state_dict = OrderedDict({item[0]: torch.zeros(item[1], dtype=torch.int32, device='cuda:0')})
        learner.model.load_state_dict(new_state_dict, strict=False)
    auc_tmp, _ = get_val_results(learner)
    per = (auc_tmp - auc_val) / auc_val * 100
    logger.info('Demographics: %.3f (%.2f)' % (auc_tmp, per))

def eval_risks(learner,bestLearner,cat_dic,cont_dic,auc_val):
    logger=logging.getLogger('importance')
    learner.load(bestLearner)
    if 'Nul' in bestLearner:
        logger.info('evaluate risks factors for nulliparous...')
        for item in [(k, v.shape) for k, v in learner.model.state_dict().items()][cat_dic['rf_pdiab']:cat_dic['rf_phype'] + 1]:
            new_state_dict = OrderedDict({item[0]: torch.zeros(item[1], dtype=torch.int32, device='cuda:0')})
            learner.model.load_state_dict(new_state_dict, strict=False)
    else:
        logger.info('evaluate risks factors for multiparous...')
        for item in [(k, v.shape) for k, v in learner.model.state_dict().items()][cat_dic['rf_pdiab']:cat_dic['rf_ppterm'] + 1]:
            new_state_dict = OrderedDict({item[0]: torch.zeros(item[1], dtype=torch.int32, device='cuda:0')})
            learner.model.load_state_dict(new_state_dict, strict=False)
    auc_tmp, _ = get_val_results(learner)
    per = (auc_tmp - auc_val) / auc_val * 100
    logger.info('risks: %.3f (%.2f)' % (auc_tmp, per))

def eval_infant(learner,bestLearner,cat_dic,cont_dic,auc_val):
    logger=logging.getLogger('importance')
    learner.load(bestLearner)
    for item in [(k, v.shape) for k, v in learner.model.state_dict().items()][cat_dic['dplural']:cat_dic['sex'] + 1]:
        new_state_dict = OrderedDict({item[0]: torch.zeros(item[1], dtype=torch.int32, device='cuda:0')})
        learner.model.load_state_dict(new_state_dict, strict=False)
    auc_tmp, _ = get_val_results(learner)
    per = (auc_tmp - auc_val) / auc_val * 100
    logger.info('Infant: %.3f (%.2f)' % (auc_tmp, per))

def eval_tobacco(learner,bestLearner,cat_dic,cont_dic,auc_val):
    logger=logging.getLogger('importance')
    learner.load(bestLearner)
    layer0 = learner.model.state_dict()['layers.0.weight']
    for i in [cont_dic['cig_0'], cont_dic['cig_1'] ]:
        layer0.index_fill_(1, torch.tensor(i).cuda(), 0)
    new_state_dict = OrderedDict({'layers.0.weight': layer0})
    learner.model.load_state_dict(new_state_dict, strict=False)
    auc_tmp, _ = get_val_results(learner)
    per = (auc_tmp - auc_val) / auc_val * 100
    logger.info('Tobacco use: %.3f (%.2f)' % (auc_tmp, per))

def eval_bmi_height(learner,bestLearner,cat_dic,cont_dic,auc_val):
    logger=logging.getLogger('importance')
    learner.load(bestLearner)
    layer0 = learner.model.state_dict()['layers.0.weight']
    for i in [cont_dic['bmi'], cont_dic['m_ht_in']]:
        layer0.index_fill_(1, torch.tensor(i).cuda(), 0)
    new_state_dict = OrderedDict({'layers.0.weight': layer0})
    learner.model.load_state_dict(new_state_dict, strict=False)
    auc_tmp, _ = get_val_results(learner)
    per = (auc_tmp - auc_val) / auc_val * 100
    logger.info('BMI & height: %.3f (%.2f)' % (auc_tmp, per))

def eval_history(learner,bestLearner,cat_dic,cont_dic,auc_val):
    logger=logging.getLogger('importance')
    learner.load(bestLearner)
    layer0 = learner.model.state_dict()['layers.0.weight']
    delete = ['illb_r', 'ilop_r', 'ilp_r', 'priorlive', 'priordead', 'priorterm']
    for i in delete:
        layer0.index_fill_(1, torch.tensor(cont_dic[i]).cuda(), 0)
        new_state_dict = OrderedDict({'layers.0.weight': layer0})
        learner.model.load_state_dict(new_state_dict, strict=False)
    for item in [(k, v.shape) for k, v in learner.model.state_dict().items()][
                cat_dic['rf_ppterm']:cat_dic['rf_ppterm'] + 1]:
        new_state_dict = OrderedDict({item[0]: torch.zeros(item[1], dtype=torch.int32, device='cuda:0')})
        learner.model.load_state_dict(new_state_dict, strict=False)
    auc_tmp, _ = get_val_results(learner)
    per = (auc_tmp - auc_val) / auc_val * 100
    logger.info('Obtetric History: %.3f (%.2f)' % (auc_tmp, per))

def eval_cont(learner,bestLearner,cat_dic,cont_dic,auc_val):
    logger=logging.getLogger('importance')
    n = (learner.model.state_dict()['layers.0.weight']).shape[1]
    n_emb = learner.model.n_emb
    cont_auc={}
    for v,i in cont_dic.items():
        learner.load(bestLearner)
        layer0=learner.model.state_dict()['layers.0.weight']
        layer0.index_fill_(1, torch.tensor(i).cuda(),0)
        new_state_dict = OrderedDict({'layers.0.weight': layer0})
        learner.model.load_state_dict(new_state_dict, strict=False)
        y=learner.get_preds(ds_type=DatasetType.Valid)
        y2=y[1]
        y1=y[0]
        Y2=1-to_np(y2)
        Y1=to_np(y1[:,0])
        auc_tmp=roc_auc_score(Y2,Y1)
        per = (auc_tmp - auc_val) / auc_val * 100
        logger.info('%s: %.3f (%.2f)' % (v, auc_tmp, per))
        cont_auc[v]=(auc_tmp,per)
    return cont_auc

def eval_cat(learner,bestLearner,cat_dic,cont_dic,auc_val,cat_names):
    logger=logging.getLogger('importance')
    n = (learner.model.state_dict()['layers.0.weight']).shape[1]
    n_emb = learner.model.n_emb
    cat_auc={}
    i=0
    for item in [(k,v.shape) for k,v in learner.model.state_dict().items()][0:len(cat_names)]:
        learner.load(bestLearner)
        new_state_dict = OrderedDict({item[0]: torch.zeros(item[1],dtype=torch.int32,device='cuda:0')})
        learner.model.load_state_dict(new_state_dict, strict=False)
        y=learner.get_preds(ds_type=DatasetType.Valid)
        y2=y[1]
        y1=y[0]
        Y2=1-to_np(y2)
        Y1=to_np(y1[:,0])
        auc_tmp=roc_auc_score(Y2,Y1)
        per = (auc_tmp - auc_val) / auc_val * 100
        logger.info('%s: %.3f (%.2f)' % (cat_names[i], auc_tmp, per))
        cat_dic[cat_names[i]]=(auc_tmp,per)
        i+=1
    return cat_dic

def eval_importance_nul(learner,bestLearner):
    # get val AUROC
    logger=logging.getLogger('importance')
    learner.load(bestLearner)
    auc_val,_=get_val_results(learner)
    logger.info('auc for validation',auc_val)

    n = (learner.model.state_dict()['layers.0.weight']).shape[1]
    n_emb = learner.model.n_emb
    cont_dic = dict(zip(con_nul, range(n_emb, n_emb + len(con_nul))))
    cat_dic = dict(zip(cat_nul, range(len(cat_nul))))
    eval_demographic(learner,bestLearner,cat_dic,cont_dic,auc_val)
    eval_risks(learner,bestLearner,cat_dic,cont_dic,auc_val)
    eval_infection(learner,bestLearner,cat_dic,cont_dic,auc_val)
    eval_tobacco(learner,bestLearner,cat_dic,cont_dic,auc_val)
    eval_bmi_height(learner,bestLearner,cat_dic,cont_dic,auc_val)


    eval_infant(learner,bestLearner,cat_dic,cont_dic,auc_val)

    cont_aucs=eval_cont(learner,bestLearner,cat_dic,cont_dic,auc_val)
    cat_aucs=eval_cat(learner,bestLearner,cat_dic,cont_dic,auc_val,cat_nul)

def eval_importance_mul(learner, bestLearner):
    # get val AUROC
    logger = logging.getLogger('importance')
    logger.info('\n------------\n\nMultiparous Women\n\n------------')
    learner.load(bestLearner)

    auc_val, _ = get_val_results(learner)
    logger.info('auc for validation',auc_val)

    n = (learner.model.state_dict()['layers.0.weight']).shape[1]
    n_emb = learner.model.n_emb
    cont_dic = dict(zip(con_mul, range(n_emb, n_emb + len(con_mul))))
    cat_dic = dict(zip(cat_mul, range(len(cat_mul))))
    eval_demographic(learner, bestLearner, cat_dic, cont_dic, auc_val)
    eval_risks(learner, bestLearner, cat_dic, cont_dic, auc_val)
    eval_infection(learner, bestLearner, cat_dic, cont_dic, auc_val)
    eval_tobacco(learner, bestLearner, cat_dic, cont_dic, auc_val)
    eval_bmi_height(learner, bestLearner, cat_dic, cont_dic, auc_val)
    eval_history(learner, bestLearner, cat_dic, cont_dic, auc_val)
    eval_infant(learner, bestLearner, cat_dic, cont_dic, auc_val)

    cont_aucs = eval_cont(learner, bestLearner, cat_dic, cont_dic, auc_val)
    cat_aucs = eval_cat(learner, bestLearner, cat_dic, cont_dic, auc_val, cat_mul)


def split_nul_mul(df):
    return df[(df['priorlive'] == 0) & (df['priordead']==0)], df[((df['priorlive'] > 0) | (df['priordead']>0)) & (df['priorlive'] != 99) & (df['priordead'] != 99)]


def define_missing(df):
    #     df[df['mbstate_rec']==3]['mbstate_rec']=np.nan
    print('mbstate_rec:', df['mbstate_rec'].unique())
    # df[df['mbrace'] == 0]['mbrace'] = np.nan
    # print('mbrace:', df['mbrace'].unique())

    df['mar_p'] = df['mar_p'].replace('^\s*$', np.nan, regex=True)
    df['mar_p'] = df['mar_p'].replace('.*U.*', 'U', regex=True)
    df['mar_p'] = df['mar_p'].replace(r'.*1.*', 1, regex=True)
    df['mar_p'] = df['mar_p'].replace(r'.*2.*', 2, regex=True)
    df['mar_p'] = df['mar_p'].replace('.*X.*', 'X', regex=True)
    print('mar_p:', df['mar_p'].unique())

    df['dmar'] = df['dmar'].replace(r'^\s*$', np.nan, regex=True)
    df['dmar'] = df['dmar'].replace(r'.*1.*', 1, regex=True)
    df['dmar'] = df['dmar'].replace(r'.*2.*', 2, regex=True)
    print('dmar:', df['dmar'].unique())

    #     df['meduc']=df['meduc'].replace(9,np.nan,regex=True)
    print('meduc:', df['meduc'].unique())

    #     df['fhisp_r']=df['fhisp_r'].replace(9,np.nan,regex=True)
    # print('fhisp_r:', df['fhisp_r'].unique())

    #     df['fracehisp']=df['fracehisp'].replace(9,np.nan,regex=True)
    #     df['fracehisp']=df['fracehisp'].replace(8,np.nan,regex=True)
    #     print('fracehisp:',df['fracehisp'].unique())

    #     df['feduc']=df['feduc'].replace(9,np.nan,regex=True)
    print('feduc:', df['feduc'].unique())

    #     df['rf_pdiab']=df['rf_pdiab'].replace('.*U.*',np.nan,regex=True)
    print('rf_pdiab:', df['rf_pdiab'].unique())

    #     df['rf_gdiab']=df['rf_gdiab'].replace('.*U.*',np.nan,regex=True)
    #     df['rf_phype']=df['rf_phype'].replace('.*U.*',np.nan,regex=True)
    #     df['rf_ghype']=df['rf_ghype'].replace('.*U.*',np.nan,regex=True)
    #     df['rf_ehype']=df['rf_ehype'].replace('.*U.*',np.nan,regex=True)
    #     df['rf_ppterm']=df['rf_ppterm'].replace('.*U.*',np.nan,regex=True)
    #     df['rf_inftr']=df['rf_inftr'].replace('.*U.*',np.nan,regex=True)
    #     df['rf_fedrg']=df['rf_fedrg'].replace('.*U.*',np.nan,regex=True)
    #     df['rf_artec']=df['rf_artec'].replace('.*U.*',np.nan,regex=True)
    #     df['wic']=df['wic'].replace('.*U.*',np.nan,regex=True)

    print('rf deal done')
    #     df['ip_gon']=df['ip_gon'].replace('.*U.*',np.nan,regex=True)
    #     df['ip_syph']=df['ip_syph'].replace('.*U.*',np.nan,regex=True)
    #     df['ip_chlam']=df['ip_chlam'].replace('.*U.*',np.nan,regex=True)
    #     df['ip_hepatb']=df['ip_hepatb'].replace('.*U.*',np.nan,regex=True)
    #     df['ip_hepatc']=df['ip_hepatc'].replace('.*U.*',np.nan,regex=True)

    #     print('ip done')
    #     df['fagecomb']=df['fagecomb'].replace(99,np.nan,regex=True)
    #     df['priorlive']=df['priorlive'].replace(99,np.nan,regex=True)
    #     df['priorterm']=df['priorterm'].replace(99,np.nan,regex=True)
    #     df['priordead']=df['priordead'].replace(99,np.nan,regex=True)
    #     df['illb_r']=df['illb_r'].replace(999,np.nan,regex=True)
    #     df['ilop_r']=df['ilop_r'].replace(999,np.nan,regex=True)
    #     df['ilp_r']=df['ilp_r'].replace(999,np.nan,regex=True)
    #     print('history done')
    df['precare']=df['precare'].replace(99,np.nan,regex=True)
    # df['previs'] = df['previs'].replace(99, np.nan, regex=True)
    df['cig_0'] = df['cig_0'].replace(99, np.nan, regex=True)
    df['cig_1'] = df['cig_1'].replace(99, np.nan, regex=True)
    #     df['cig_2']=df['cig_2'].replace(99,np.nan,regex=True)
    #     df['cig_3']=df['cig_3'].replace(99,np.nan,regex=True)
    #     print('cigarate done')
    df['m_ht_in'] = df['m_ht_in'].replace(99, np.nan, regex=True)
    df['bmi'] = df['bmi'].replace(99.9, np.nan, regex=True)
    # df['wtgain'] = df['wtgain'].replace(99, np.nan, regex=True)
    df['rf_cesarn'] = df['rf_cesarn'].replace(99, np.nan, regex=True)
    #     print('all missing done')

    return df


def recode(df, column_list):
    for col in column_list:
        df[col] = df[col].replace(r'.*X.*', 0, regex=True)
        df[col] = df[col].replace(r'.*Y.*', 1, regex=True)
        df[col] = df[col].replace(r'.*N.*', 2, regex=True)
        df[col] = df[col].replace(r'.*F.*', 0, regex=True)
        df[col] = df[col].replace(r'.*M.*', 1, regex=True)
        print(col, df[col].unique())
    return df
00

# recode the outcome
def preterm_recode(df):
    if 'oegest_comb' in df.columns:
        df.loc[df['oegest_comb'] < 37, 'oegest_comb'] = 0
        df.loc[df['oegest_comb'] >= 37, 'oegest_comb'] = 1
        df = df.rename(columns={'oegest_comb': 'cat3'})
    else:
        if max(df['cat3']) > 37:
            df.loc[df['cat3'] < 37, 'cat3'] = 0
            df.loc[df['cat3'] >= 37, 'cat3'] = 1
    return df


def select_useful(nul, mul):
    nul_cols = ['mbstate_rec',
                # 'mbrace',
                # 'mhisp_r',
                'mracehisp',
                'mar_p',
                'dmar',
                'meduc',
                   'fracehisp',
                'feduc',
                'wic',
                'rf_pdiab',
                # 'rf_gdiab',
                'rf_phype',
                # 'rf_ppterm',
                'rf_inftr',
                'rf_fedrg',
                'rf_artec',
                'ip_gon',
                'ip_syph',
                'ip_chlam',
                'ip_hepatb',
                'ip_hepatc',
                'dplural',
                'sex',
                'mager',
                # 'fagecomb',
                'precare',
                'cig_0',
                'cig_1',
                'm_ht_in',
                'bmi',
                # 'rf_cesarn',
                'cat3'
                ]
    mul_cols = ['mbstate_rec',
                # 'mbrace',
                # 'mhisp_r',
                'mracehisp',

                'mar_p',
                'dmar',
                'meduc',
                'fracehisp',
                'feduc',
                'wic',
                'rf_pdiab',
                'rf_phype',
                'rf_ppterm',
                'rf_inftr',
                'rf_fedrg',
                'rf_artec',
                'ip_gon',
                'ip_syph',
                'ip_chlam',
                'ip_hepatb',
                'ip_hepatc',
                'dplural',
                'sex',
                'mager',
                #     'fagecomb',
                'priorlive',
                'priordead',
                'priorterm',
                'illb_r',
                'ilop_r',
                'ilp_r',
                'precare',
                'cig_0',
                'cig_1',
                'm_ht_in',
                'bmi',
                'rf_cesarn',
                'cat3'
                ]
    nul_after = nul.reindex(columns=nul_cols)
    mul_after = mul.reindex(columns=mul_cols)
    return nul_after, mul_after

