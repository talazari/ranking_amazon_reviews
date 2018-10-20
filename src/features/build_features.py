"""
Used to build features for the reviews dataset
Author: Tal Azaria
"""
import pickle
import string
import numpy as np
import pandas as pd

from tqdm import tqdm
from nltk import pos_tag
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def count_uppercase(text):
    return sum(1 for char in text if char.isupper())


def count_lowercase(text):
    return sum(1 for char in text if char.islower())


def count_punctuation(text):
    return sum([1 for char in text if char in string.punctuation])


def count_words(text):
    tokenizer = RegexpTokenizer(r'\w+')
    return len(tokenizer.tokenize(text))


def count_dots(text):
    return sum([1 for char in text if char == '.'])


def count_exclamation_marks(text):
    return sum([1 for char in text if char == '!'])


def count_question_marks(text):
    return sum([1 for char in text if char == '?'])


def count_digits(text):
    return sum([1 for char in text if char.isdigit()])


def count_stop_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    return sum([1 for word in word_tokens if word in stop_words])


# def count_sentiment_words(text, threshold=0.5):
#     """
#     Counts the number of words with a sentiment score above a threshold
#
#     Args:
#         text(str): a text
#         threshold(float): defines the threshold for sentiment score
#
#     Returns:
#
#     """
#     word_tokens = word_tokenize(text)
#     pos_word_list = []
#     neu_word_list = []
#     neg_word_list = []
#     sentence_length = len(word_tokens)
#     sid = SentimentIntensityAnalyzer()
#
#     for word in word_tokens:
#         if (sid.polarity_scores(word)['compound']) >= threshold:
#             pos_word_list.append(word)
#
#         elif (sid.polarity_scores(word)['compound']) <= -threshold:
#             neg_word_list.append(word)
#
#         else:
#             neu_word_list.append(word)
#
#     return len(pos_word_list)/sentence_length, len(neg_word_list)/sentence_length, len(neu_word_list)/sentence_length


def sentiment_analysis(text):
    """
    Calculate the sentiment of a sentence using Sentiment Analyser

    Args:
        text(str): a text

    Returns
        tuple. with the sentiment analysis score ('neg', 'neu', 'pos', 'compound')
    """
    sid = SentimentIntensityAnalyzer()
    sentiment_analysis_scores = sid.polarity_scores(text)
    return tuple(sentiment_analysis_scores.values())


def calculate_lexical_diversity(text):
    """
    Calculate the diversity of words in a sentence

    Args:
        text(str): a text

    Returns:
        float. the ratio of diversity [0,1], where 1 is all words are unique
    """
    word_tokens = word_tokenize(text)
    return len(set(word_tokens))/ len(word_tokens)


# def most_common_pos(text):
#     """
#     Find the most common part of speech in the text
#
#     Args:
#         text(str): a text
#
#     Returns:
#
#     """
#     pos = pos_tag(word_tokenize(text))
#     return Counter([x[1] for x in pos]).most_common(3)


def extract_features(reviews):
    """
    Extract features

    Args:
        reviews(pd.DataFrame):

    Returns:
        pd.DataFrame. the dataset with all the extracted features
    """
    eps = np.finfo(float).eps
    tqdm.pandas(desc='getting words count: ')

    reviews['review_words_count'] = reviews['Text'].progress_apply(count_words)
    reviews['summary_words_count'] = reviews['Summary'].progress_apply(count_words)

    tqdm.pandas(desc='getting stop words count: ')
    reviews['review_stop_words_count'] = reviews['Text'].progress_apply(count_stop_words)

    tqdm.pandas(desc='getting uppercase char count: ')
    reviews['review_uppercase_count'] = reviews['Text'].progress_apply(count_uppercase)
    reviews['summary_uppercase_count'] = reviews['Summary'].progress_apply(count_uppercase)

    tqdm.pandas(desc='getting lowercase char count: ')
    reviews['review_lowercase_count'] = reviews['Text'].progress_apply(count_lowercase)
    reviews['summary_lowercase_count'] = reviews['Summary'].progress_apply(count_lowercase)

    tqdm.pandas(desc='getting dot char count: ')
    reviews['review_dots_count'] = reviews['Text'].progress_apply(count_dots)
    reviews['summary_dots_count'] = reviews['Summary'].progress_apply(count_dots)

    tqdm.pandas(desc='getting exclamation mark char count: ')
    reviews['review_exclamation_mark_count'] = reviews['Text'].progress_apply(count_exclamation_marks)
    reviews['summary_exclamation_mark_count'] = reviews['Summary'].progress_apply(count_exclamation_marks)

    tqdm.pandas(desc='getting question mark char count: ')
    reviews['review_question_mark_count'] = reviews['Text'].progress_apply(count_question_marks)
    reviews['summary_question_mark_count'] = reviews['Summary'].progress_apply(count_question_marks)

    tqdm.pandas(desc='getting punctuation char count: ')
    reviews['review_punctuation_count'] = reviews['Text'].progress_apply(count_punctuation)
    reviews['summary_punctuation_count'] = reviews['Summary'].progress_apply(count_punctuation)

    tqdm.pandas(desc='getting digit char count: ')
    reviews['review_digits_count'] = reviews['Text'].progress_apply(count_digits)
    reviews['summary_digits_count'] = reviews['Summary'].progress_apply(count_digits)

    tqdm.pandas(desc='calculating lexical diversity: ')
    reviews['review_lexical_diversity'] = reviews['Text'].progress_apply(calculate_lexical_diversity)
    reviews['summary_lexical_diversity'] = reviews['Summary'].progress_apply(calculate_lexical_diversity)

    print('calculating upper_lowercase ratio: ')
    reviews['review_upper_lower_case_ratio'] = reviews['review_uppercase_count'] / (reviews['review_lowercase_count'] + eps)
    reviews['summary_upper_lower_case_ratio'] = reviews['summary_uppercase_count'] / (reviews['summary_lowercase_count'] + eps)

    print('getting sentences count:  ')
    reviews['review_sentences_count'] = reviews['review_uppercase_count'] / (reviews['review_dots_count'] + eps)
    reviews['summary_sentences_count'] = reviews['summary_uppercase_count'] / (reviews['summary_dots_count'] + eps)

    print('calculating the review/summary uppercase ratio: ')
    reviews['review_summary_uppercase_ratio'] = reviews['review_uppercase_count'] / (reviews['summary_uppercase_count'] + eps)

    tqdm.pandas(desc='getting reviews sentiment analysis: ')
    reviews[['neg', 'neu', 'pos', 'compound']] = reviews['Text'].agg(sentiment_analysis).progress_apply(pd.Series)
    print('calculating log of features: ')
    reviews['log_product_frequency'] = np.log2(reviews['ProductFrequency'])
    reviews['log_user_reviews_count'] = np.log2(reviews['user_reviews_count'])
    reviews['log_review_words_count'] = np.log2(reviews['review_words_count'])

    print('getting timestamp information: ')
    reviews['timestamp'] = pd.to_datetime(reviews.Time, unit='s')
    reviews['year'] = reviews.timestamp.dt.year
    reviews['month'] = reviews.timestamp.dt.month
    reviews['day'] = reviews.timestamp.dt.day

    with open('data/interim/reviews_post_feature_eng.pkl', 'wb') as f:
        pickle.dump(reviews, f)

    return reviews

