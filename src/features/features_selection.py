"""
Used to select the most significant features for modeling
Author: Tal Azaria
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)


def eliminate_features(x, y, n_features):
    """
    Eliminate features from dataset

    Args:
        x(pd.DataFrame): dataset's features
        y(pd.Series): dataset's labels
        n_features(int): the required number of features

    Returns:
        np.array: features mask
    """
    estimator = RandomForestClassifier()
    selector = RFE(estimator, n_features)
    selector = selector.fit(x, y)
    return selector.support_


def get_accuracy_scores(x, y, features_mask):
    """
    Split the dataset to train test and validation, eliminate features by a mask and returns the accuracy score
    for random forest classifier

    Args:
       X(pd.DataFram): dataset's features
       y(pd.Series): dataset's labels
       features_mask(np.array): the required number of features


    Returns:
        (float): accuracy score for linear regression model
    """
    x = x.loc[:, features_mask]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    return accuracy_score(y_val, y_pred)


def select_features_by_recursive_elimination(x, y, original_feature_no):
    """
    Filter features using recursive feature elimination method (increase gradually the number of features)

    Args:
        x(pd.DataFram): dataset's features
        y(pd.Series): dataset's labels
        original_feature_no(int): the original number of features

    Returns:
        tuple. accuracy scores array and feature mask array
    """
    accuracy_scores = []
    feature_masks = []
    feature_no = np.arange(1, original_feature_no + 1)

    for n in tqdm(feature_no):
        features_mask = eliminate_features(x, y, n)
        accuracy_scores.append(get_accuracy_scores(x, y, features_mask))
        feature_masks.append(features_mask)
    return accuracy_scores, feature_masks


def select_features_with_max_accuracy(reviews):
    """
    Select and save the feature mask which give the maximum accuracy score

    Args:
        reviews(pd.DataFrame): the dataset

    Returns:
        pd.DataFrame: with the selected features
    """
    label = ['HelpfulnessRank']
    x_temp = reviews.drop(label, axis=1)
    textual_and_label_features = ['Text', 'Summary', 'timestamp', 'ProfileName', 'ProductId', 'UserId',
                                  'HelpfulnessNumerator', 'HelpfulnessDenominator']
    x = x_temp.drop(textual_and_label_features, axis=1)
    y = reviews[label]
    original_feature_no = x.shape[1]
    accuracy_scores, feature_masks = select_features_by_recursive_elimination(x, y, original_feature_no)

    # The number of feature which result with the maximum validation accuracy:
    ind_max = accuracy_scores.index(max(accuracy_scores))
    selected_feature_max = feature_masks[ind_max]  # the relevant feature mask
    x_ = x.loc[:, selected_feature_max]
    reviews = pd.concat([reviews[x_.columns], y], axis=1)

    with open(os.path.join('data', 'interim', 'reviews_post_feature_selection.pkl'), 'wb') as f:
        pickle.dump(reviews, f)

    return reviews
