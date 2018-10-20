"""
Used to train models and predict the helpfulness rank of reviews
Author: Tal Azaria
"""

import os
import pickle

import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureEngineerModel(object):
    """ Holds functionality for training an sv,m. random forest or xgboost models"""
    def __init__(self, dataset, model_type, kwargs):
        self.reviews = dataset
        self.model_type = model_type
        name_to_model = {'svm': SVC(),
                         'xgboost': XGBClassifier(),
                         'random_forest': RandomForestClassifier()}

        self.model = name_to_model[model_type]

        for parameter, value in kwargs.items():
            setattr(self.model, parameter, value)

        # TODO - Ask Noa if it is acceptable
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = \
            self.train_validation_test_split()

    def train_validation_test_split(self):
        """
        Split the data set into train, validation and test, and change its format according to the model type

        Returns:
            tuple. of pd.DataFrames with the dataset features and labels
        """
        label = ['HelpfulnessRank']
        x = self.reviews.drop(label, axis=1)
        y = self.reviews[label]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

        if self.model_type == 'xgboost':
            x_train = x_train.astype('float64')
            x_test = x_test.astype('float64')
            x_val = x_val.astype('float64')

        elif self.model_type == 'svm':
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.fit_transform(x_test)
            x_val = scaler.fit_transform(x_val)

        return x_train, x_val, x_test, y_train, y_val, y_test

    def save_model(self):
        model_path = 'models/' + self.model_type + '_fe.pk1'
        joblib.dump(self.model, model_path)

    @staticmethod
    def save_predictions(model_type, dataset, prediction, proba):
        prediction_path = 'models/' + model_type + '_fe_' + dataset + '_pred.npy'
        np.save(prediction_path, prediction)

        proba_path = 'models/' + model_type + '_fe_' + dataset + '_proba.pkl'
        with open(proba_path, 'wb') as f:
            pickle.dump(proba, f)

    def train_and_predict(self):
        """Train, predict and save predictions"""
        datasets = {'train': self.x_train,
                    'test': self.x_test,
                    'validation': self.x_val}

        labels = {'train': self.y_train,
                  'test': self.y_test,
                  'validation': self.y_val}

        print("Train")
        print("------------------------------------")
        self.model.fit(self.x_train, self.y_train)

        print("Inference")
        print("------------------------------------")

        for split in datasets:
            prediction = self.model.predict(datasets[split])
            proba = self.model.predict_proba(datasets[split])
            accuracy = accuracy_score(labels[split], prediction)
            print('{} accuracy score: {}'.format(split, accuracy))
            print("Saving Predictions")
            self.save_predictions(self.model_type, split, prediction, proba)

        self.save_model()


class TrainTextualModel(object):
    """ Holds functionality for training an svm, random forest or xgboost models using textual vectorizers"""
    def __init__(self, dataset, model_type, vectorizer_type, kwargs):
        self.reviews = dataset
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type

        x = self.reviews.drop('HelpfulnessRank', axis=1)
        y = self.reviews['HelpfulnessRank']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.4, random_state=42)
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(self.x_test, self.y_test, test_size=0.5,
                                                                            random_state=42)

        name_to_classifier = {'random_forest': RandomForestClassifier(),
                              'xgboost': XGBClassifier(),
                              'svm': SVC()}

        name_to_vectorizer = {'bag_of_words': CountVectorizer(min_df=5,  ngram_range=(1, 1)),
                              'tfidf': TfidfVectorizer(min_df=5,  ngram_range=(1, 1))}

        classifier = name_to_classifier[model_type]
        vectorizer = name_to_vectorizer[vectorizer_type]

        for parameter, value in kwargs.items():
            setattr(classifier, parameter, value)

        self.pipeline = Pipeline([('vectorizer', vectorizer),
                                  ('classifier', classifier)])

    def save_model(self):
        model_path = os.path.join('models', self.model_type + self.vectorizer_type + '.pk1')
        joblib.dump(self.pipeline, model_path)

    @staticmethod
    def save_predictions(model_type, vectorizer_type, dataset, prediction, proba):
        prediction_path = os.path.join('models', model_type + '_' + vectorizer_type + '_' + dataset + '_pred.npy')
        np.save(prediction_path, prediction)

        proba_path = os.path.join('models', model_type + '_' + vectorizer_type + '_' + dataset + '_proba.pkl')
        with open(proba_path, 'wb') as f:
            pickle.dump(proba, f)

    def train_and_predict(self):
        """
        Train the model and predict the accuracy score and save the model, the prediction and the probabilities
        """
        train = list(zip(self.x_train['Text'], self.y_train.values))
        val = list(zip(self.x_val['Text'], self.y_val.values))
        test = list(zip(self.x_test['Text'], self.y_test.values))

        datasets = {'train': train,
                    'test': val,
                    'validation': test}

        labels = {'train': self.y_train,
                  'test': self.y_test,
                  'validation': self.y_val}

        print("Train")
        print("------------------------------------")
        self.pipeline.fit([x[0] for x in train], [x[1] for x in train])
        print("Inference")
        print("------------------------------------")

        for dataset in datasets:
            prediction = self.pipeline.predict([x[0] for x in datasets[dataset]])
            proba = self.pipeline.predict_proba([x[0] for x in datasets[dataset]])
            accuracy = accuracy_score(labels[dataset], prediction)
            print('{} accuracy score: {}'.format(dataset, accuracy))
            print("Saving Predictions")
            self.save_predictions(self.model_type, self.vectorizer_type, dataset, prediction, proba)

        self.save_model()

# TODO - vectorizer parameter, realize what was actually run
