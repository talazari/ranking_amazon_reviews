"""
Rank reviews by their helpfulness
Author: Tal Azaria
"""
import argparse
import json
import os

import pandas as pd
from tqdm import tqdm

from features.build_features import extract_features
from data.preprocess_dataset import pre_process_dataset
from features.features_selection import select_features_with_max_accuracy
from models.train_and_predict import TrainTextualModel, FeatureEngineerModel
from models.evaluation import evaluate_models_ranking_predictions
from models.ensemble_and_stacking import load_probas, ensemble_models_and_evaluate_accuracy, \
    stack_models_and_evaluate_accuracy


# TODO - docstring for file and func documentation


def rank_reviews_by_helpfulness(hyperparams):
    """ Load, pre-process the reviews dataset, engineer features, train and predict the reviews helpfulness

    Args:
        hyperparams(dict): consist of the model's hyper-parameters for training

    * The Reviews dateset is load, it goes though pre-processing and feature engineering
    * The most important features are selected, along with textual features  which are extracted using tfidf and bow
    * Three types of models are trained: Random forest, XGboost and SVM either with the selected features
    * or with the textual features
    * In order to improve the performance the following was applied:
    *               * Ensembled methods (equal and weighted)
    *               * Stacking methods (Logistic Regression and Random Forest
    * The predictions are saved into files
    * The predictions, aka the reviews helpfulness ranking scores are evaluated using Kendall Tau distance
    """
    reviews = pd.read_csv(os.path.join('data', 'raw', 'Reviews.csv'))
    reviews = pre_process_dataset(reviews)
    reviews = extract_features(reviews)
    reviews_textual_features = reviews[['Text', 'Summary', 'HelpfulnessRank']]
    reviews_selected_features = select_features_with_max_accuracy(reviews)

    # Models parameters were chosen after optimization (notebook no. 5)
    rnd_forest_feature_eng = FeatureEngineerModel(dataset=reviews_selected_features,
                                                  model_type='random_forest',
                                                  kwargs=hyperparams['feature_engineering']['random_forest'])

    svm_feature_eng = FeatureEngineerModel(dataset=reviews_selected_features,
                                           model_type='svm',
                                           kwargs=hyperparams['feature_engineering']['svm'])

    xgboost_feature_eng = FeatureEngineerModel(dataset=reviews_selected_features,
                                               model_type='xgboost',
                                               kwargs=hyperparams['feature_engineering']['xgboost'])

    rnd_forest_bow = TrainTextualModel(dataset=reviews_textual_features,
                                       model_type='random_forest',
                                       vectorizer_type='bag_of_words',
                                       kwargs=hyperparams['textual_features']['random_forest'])

    rnd_forest_tfidf = TrainTextualModel(dataset=reviews_textual_features,
                                         model_type='random_forest',
                                         vectorizer_type='tfidf',
                                         kwargs=hyperparams['textual_features']['random_forest'])

    svm_forest_bow = TrainTextualModel(dataset=reviews_textual_features,
                                       model_type='svm',
                                       vectorizer_type='bag_of_words',
                                       kwargs=hyperparams['textual_features']['svm'])

    svm_tfidf = TrainTextualModel(dataset=reviews_textual_features,
                                  model_type='svm',
                                  vectorizer_type='tfidf',
                                  kwargs=hyperparams['textual_features']['svm'])

    xgboost_forest_bow = TrainTextualModel(dataset=reviews_textual_features,
                                           model_type='xgboost',
                                           vectorizer_type='bag_of_words',
                                           kwargs=hyperparams['textual_features']['xgboost'])

    xgboost_forest_tfidf = TrainTextualModel(dataset=reviews_textual_features,
                                             model_type='xgboost',
                                             vectorizer_type='tfidf',
                                             kwargs=hyperparams['textual_features']['xgboost'])

    for model in tqdm([rnd_forest_feature_eng, svm_feature_eng, xgboost_feature_eng,
                       rnd_forest_bow, rnd_forest_tfidf, svm_forest_bow, svm_tfidf,
                       xgboost_forest_bow, xgboost_forest_tfidf]):
        model.train_and_predict()

    train_probas, val_probas, test_probas = load_probas()
    ensemble_models_and_evaluate_accuracy(train_probas=train_probas, val_probas=val_probas, test_probas=test_probas,
                                          y_train=rnd_forest_feature_eng.y_train, y_val=rnd_forest_feature_eng.y_val,
                                          y_test=rnd_forest_feature_eng.y_test)

    stack_models_and_evaluate_accuracy(train_probas=train_probas, val_probas=val_probas, test_probas=test_probas,
                                       y_train=rnd_forest_feature_eng.y_train, y_val=rnd_forest_feature_eng.y_val,
                                       y_test=rnd_forest_feature_eng.y_test)

    evaluate_models_ranking_predictions(reviews)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking Reviews by their Helpfulness')
    parser.add_argument('--config_file', default='src/models/config_file.json',
                        type=str, help='a path to *.json file with the models configuration')

    args = parser.parse_args()

    with open(os.path.join('src', 'models', args.congig_file)) as f:
        models_hyperparams = json.load(f)

    rank_reviews_by_helpfulness(models_hyperparams)
