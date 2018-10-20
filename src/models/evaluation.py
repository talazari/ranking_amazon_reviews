"""
Holds the functionality for evaluating the ranking using Kandall Tau distance
Author: Tal Azaria
"""

import os
import glob

import itertools
import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import kendalltau, weightedtau
from sklearn.model_selection import train_test_split


def generate_ranking_predictions():
    """
    Generate a dictionary of the used model's predictions

    Returns:
        dict. of the model name and the object of the predictions
    """
    predictions = {}
    predictions_file_names = [f for f in glob.glob(os.path.join("models", "*.npy")) if "_pred" in f]
    for filename in predictions_file_names:
        model_name = filename.rsplit('.')[0].split('/')[1].strip('_pred')
        predictions[model_name] = np.load(filename)

    return predictions


def kendall_tau_dist_norm(A, B):
    """
    Calculate the normalized Kendall tau distance

    Args:
        A(list): Ranked list, 1
        B(list): Ranked list, 2

    Returns:
        float. The normalized Kendall distance
    """
    pairs = itertools.combinations(range(0, len(A)), 2)

    distance = 0

    for x, y in pairs:
        a = A[x] - A[y]
        b = B[x] - B[y]

        # if discordant (different signs)
        if a * b < 0:
            distance += 1

    return distance / comb(len(A), 2)


def evaluate_product_ranking(products_data, method):
    """
    Evaluate the predictions ranking with respect to the ground truth rank

    Args:
        products_data(pandas.core.groupby.DataFrameGroupBy): The reviews grouped by ProductId
        method(function): The Evaluation method either

    Returns:
        float. The score for evaluation of ranking
    """
    # Order by Helpfulness score and index and get ranked indices:
    prediction = products_data.groupby('Predicted_Helpfulness_Score').apply(pd.DataFrame.sortlevel, level=0,
                                                                             ascending=True)

    prediction.sort_index(ascending=False, level=0, inplace=True)
    prediction_rank = prediction.index.labels[1]
    ground_truth = products_data.groupby('Ground_Truth_Helpfulness_Score').apply(pd.DataFrame.sortlevel, level=0,
                                                                                 ascending=True)

    ground_truth.sort_index(ascending=False, level=0, inplace=True)
    ground_truth_rank = ground_truth.index.labels[1]

    return method(prediction_rank, ground_truth_rank)


def evaluate_ranking_predictions(x_val, y_val, y_pred, method):
    """
    Calculate the ranking evaluation for all products

    Args:
        x_val(pd.DataFrame): The Data set for evaluation
        y_val(pd.Series): The Data set labels (ground truth)
        y_pred(np.ndarray): The label's prediction
        method(func): The method for ranking evaluation

    Returns:
        pd.DataFrame: With the ranking evaluation score of all the products with more than 1 review
    """
    # Add columns of the predicted and ground truth helpfulness score:
    x_val['Predicted_Helpfulness_Score'] = pd.Series(y_pred, index=x_val.index)
    x_val['Ground_Truth_Helpfulness_Score'] = pd.Series(y_val, index=x_val.index)

    # Group by ProductId and Evaluate ranking:
    rank_evaluation = x_val.groupby('ProductId').apply(evaluate_product_ranking, method=method)
    rank_evaluation = rank_evaluation.apply(pd.Series)
    rank_evaluation = rank_evaluation.rename(columns={0: 'kendall_tau', 1: 'p_value'})

    # Drop all products who had only one review:
    rank_evaluation.dropna(how='all', inplace=True)

    return rank_evaluation, rank_evaluation.mean()


def evaluate_models_ranking_predictions(reviews):
    """
    Evaluate the ranking for each one of the used models

    Args:
        reviews(pd.DataFrame): the dataset
        predictions(dict): a dict of the models and their predictions

    """
    predictions = generate_ranking_predictions()
    label = ['HelpfulnessRank']
    x = reviews.drop(label, axis=1)
    y = reviews[label]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    for key, value in predictions.items():
        _, rank_score = evaluate_ranking_predictions(x_val=x_val, y_val=y_val, y_pred=value,
                                                     method=kendall_tau_dist_norm)
        print("Model's ranking evaluation: ")
        print("----------------------------")
        print(key, rank_score)


