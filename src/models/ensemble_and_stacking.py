"""
Holds functionality which allows to ensemble and stack models
Author: Tal Azaria
"""

import os
import glob
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def load_probas():
    """
    Load prediction probabilities for all the models from files

    Returns:
        tuple. lists consisting the train, validation and test prediction probabilities
    """
    proba_file_names = [f for f in glob.glob(os.path.join("models", "*.pkl")) if "_proba" in f]

    train_probas = []
    test_probas = []
    val_probas = []

    for filename in proba_file_names:
        with open(filename, 'rb') as f:
            proba = pickle.load(f)

        if 'train' in filename:
            train_probas.append(proba)

        elif 'test' in filename:
            test_probas.append(proba)

        elif 'validation' in filename or 'val' in filename:
            val_probas.append(proba)

    return train_probas, val_probas, test_probas


def retrieve_prediction(proba):
    """
    Gets a list of prediction probabilities matrices and calculate their confidence values

    Args:
        proba(np.array): the predictions probability matrices (np.array)

    Returns:
        (np.array). a list of confidence values (accuracy score)
    """

    return proba.argmax(axis=1)


def stack_probas(predict_probas):
    """
    Stack the prediction probabilities

    Args:
        predict_probas(list): a list of train/val/test prediction probabilites from several models

    Returns:
        np.array. with all the probabilities concatenated
    """
    for ind, curr_predic_proba in enumerate(predict_probas):
        if ind == 0:
            stacked_proba = curr_predic_proba
        else:
            stacked_proba = np.concatenate((stacked_proba, curr_predic_proba), axis=1)
    return stacked_proba


def equally_ensemble_results(predictions_probas):
    """
    Ensemble prediction result with equal weight

    Args:
        predictions_probas(list): A list of the predictions probability matrices (np.array)

    Returns:
        np.array. The predictions for the equally ensembled models
    """
    sum_predictions_prob = np.zeros(predictions_probas[0].shape)

    for curr_pred in predictions_probas:
        sum_predictions_prob += curr_pred

    ensemble_prob = np.divide(sum_predictions_prob, len(predictions_probas))

    return np.argmax(ensemble_prob, axis=1)


def calculate_confidence_val(predictions_probas, y):
    """
    Get a list of prediction probabilities matrices and calculate their confidence values

    Args:
         predictions_probas(list): A list of the predictions probability matrices (np.array)
         y(np.array): labels (ground truth)
    Returns:
        (list). a list of confidence values (accuracy score)
    """
    confidence_values = []

    for proba in predictions_probas:
        prediction = proba.argmax(axis=1)
        confidence_values.append(accuracy_score(y, prediction))

    return confidence_values


def weighted_ensemble_results(predictions_probas, confidence_values):
    """
    Ensemble prediction result with weight according to each model confidence value

    Args:
        predictions_probas(list): A list of the predictions probability matrices (np.array)
        confidence_values(list): A list of confidence values (accuracy score)
    Returns:
        np.array. The predictions for the equally ensembled models
    """
    sum_predictions_prob = np.zeros(predictions_probas[0].shape)

    for curr_pred, w in zip(predictions_probas, confidence_values):
        sum_predictions_prob += w * curr_pred

    ensemble_prob = np.divide(sum_predictions_prob, sum(confidence_values))

    return np.argmax(ensemble_prob, axis=1)


def stack_models(train_probas, val_probas, y_train, y_val, model, test_probas=None, y_test=None):
    """
    Stack model's predictions

    Args:
        train_probas(list): train dataset probabilities (used as features) from all models
        val_probas(list): validation dataset probabilities from all models
        y_train(pd.Series): train dataset labels, helpfulness rank
        y_val(pd.Series): validation dataset labels, helpfulness rank
        model(object): the model's object which use for stacking
        test_probas(list): test dataset probabilities from all models
        y_test(pd.Series): test dataset labels, helpfulness rank

    Returns:
         tuple. with the train validation and test accuracy and predictions
    """

    stacked_proba_train = stack_probas(train_probas)
    stacked_proba_val = stack_probas(val_probas)
    model.fit(stacked_proba_train, y_train)
    stacked_train_pred = model.predict(stacked_proba_train)
    stacked_val_pred = model.predict(stacked_proba_val)
    train_accuracy = accuracy_score(stacked_train_pred, y_train)
    val_accuracy = accuracy_score(stacked_val_pred, y_val)
    if y_test.any():
        stacked_proba_test = stack_probas(test_probas)
        stacked_test_pred = model.predict(stacked_proba_test)
        test_accuracy = accuracy_score(y_test, stacked_test_pred)

    else:
        test_accuracy = None

    return (train_accuracy, val_accuracy, test_accuracy), stacked_train_pred, stacked_val_pred, stacked_test_pred


def ensemble_models_and_evaluate_accuracy(train_probas, val_probas, test_probas, y_train, y_val, y_test):
    """
    Ensemble the models (equally and weighted) and calculate the accuracy scores of the ensemble models

    Args:
        train_probas(list): train dataset prediction probabilities from all models
        val_probas(list): validation dataset prediction probabilities from all models
        test_probas(list): test dataset prediction probabilities from all models
        y_train(pd.Series): train dataset labels , helpfulness_rank
        y_val(pd.Series): validation dataset labels , helpfulness_rank
        y_test(pd.Series): test dataset labels , helpfulness_rank
    """
    train_eq_ensemble_pred = equally_ensemble_results(train_probas)
    val_eq_ensemble_pred = equally_ensemble_results(val_probas)
    test_eq_ensemble_pred = equally_ensemble_results(test_probas)

    print("Equally weighted ensemble:")
    print("--------------------------")
    print("Train accuracy: ", accuracy_score(y_train, train_eq_ensemble_pred))
    print("Validation accuracy: ", accuracy_score(y_val, val_eq_ensemble_pred))
    print("Test accuracy: ", accuracy_score(y_test, test_eq_ensemble_pred))

    np.save(os.path.join('model', 'train_eq_ensemble_pred'), train_eq_ensemble_pred)
    np.save(os.path.join('model', 'val_eq_ensemble_pred'), val_eq_ensemble_pred)
    np.save(os.path.join('model', 'test_eq_ensemble_pred'), test_eq_ensemble_pred)

    confidence_train = calculate_confidence_val(train_probas, y_train)
    confidence_val = calculate_confidence_val(val_probas, y_val)
    confidence_test = calculate_confidence_val(test_probas, y_test)

    train_w_ensemble_pred = weighted_ensemble_results(train_probas, confidence_train)
    val_w_ensemble_pred = weighted_ensemble_results(val_probas, confidence_val)
    test_w_ensemble_pred = weighted_ensemble_results(test_probas, confidence_test)

    print("Weighted ensemble:")
    print("--------------------------")
    print("Train accuracy: ", accuracy_score(y_train, train_w_ensemble_pred))
    print("Validation accuracy: ", accuracy_score(y_val, val_w_ensemble_pred))
    print("Test accuracy: ", accuracy_score(y_test, test_w_ensemble_pred))

    np.save(os.path.join('model', 'train_w_ensemble_pred.npy'), train_w_ensemble_pred)
    np.save(os.path.join('model', 'val_w_ensemble_pred.npy'), val_w_ensemble_pred)
    np.save(os.path.join('model', 'test_w_ensemble_pred.npy'), test_w_ensemble_pred)


def stack_models_and_evaluate_accuracy(train_probas, val_probas, test_probas, y_train, y_val, y_test):
    """
    Stack the models and calculate the accuracy scores of the stacked models using random forest and logistic regression

    Args:
        train_probas(list): train dataset prediction probabilities from all models
        val_probas(list): validation dataset prediction probabilities from all models
        test_probas(list): test dataset prediction probabilities from all models
        y_train(pd.Series): train dataset labels , helpfulness_rank
        y_val(pd.Series): validation dataset labels , helpfulness_rank
        y_test(pd.Series): test dataset labels , helpfulness_rank

    """
    logreg = LogisticRegression()
    rfc = RandomForestClassifier(n_estimators=200, max_depth=20)

    print("Stacking using Random Forest:")
    print("-----------------------------")
    stacking_accuracy_logreg, train_stack_logreg_pred, val_stack_logreg_pred, test_stack_logreg_pred = \
        stack_models(train_probas, val_probas, y_train, y_val, logreg, test_probas, y_test)
    print("train, validation and test accuracy scores:", stacking_accuracy_logreg)

    print("Stacking using Logistic Regression:")
    print("-----------------------------------")
    stacking_accuracy_rfc, train_stack_rfc_pred, val_stack_rfc_pred, test_stack_rfc_pred = \
        stack_models(train_probas, val_probas, y_train, y_val, rfc, test_probas, y_test)
    print("train, validation and test accuracy scores:", stacking_accuracy_rfc)

    np.save(os.path.join('models', 'train_stack_logreg_pred.npy'), train_stack_logreg_pred)
    np.save(os.path.join('models', 'val_stack_logreg_pred.npy'), val_stack_logreg_pred)
    np.save(os.path.join('models', 'test_stack_logreg_pred.npy'), test_stack_logreg_pred)

    np.save(os.path.join('models', 'train_stack_rfc_pred.npy'), train_stack_logreg_pred)
    np.save(os.path.join('models', 'val_stack_rfc_pred.npy'), val_stack_logreg_pred)
    np.save(os.path.join('models', 'test_stack_rfc_pred.npy'), test_stack_logreg_pred)



