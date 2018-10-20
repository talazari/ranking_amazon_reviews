"""
Used to apply pre-processing of the reviews dataset DataFrame
Author: Tal Azaria
"""
import os
import re
import spacy
import pickle
import regex as re
import numpy as np
import pandas as pd

from nltk import TweetTokenizer
from spacy.lang.en import English


DOWN_SAMPLE_RATIO = 0.25
HELPFULNESS_RANK = 10
MIN_REVIEWS_PER_PRODUCT = 10
APOSTROPHE_LOOKUP = {"aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not",
                     "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                     "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "i'd": "I would",
                     "i'll": "I will", "i'm": "I am", "isn't": "is not", "it's": "it is", "it'll": "it will",
                     "i've": "I have", "let's": "let us", "mightn't": "might not", "mustn't": "must not",
                     "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
                     "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
                     "they'll": "they will", "they're": "they are", "they've": "they have", "we'd": "we would",
                     "we're": "we are", "weren't": "were not", "we've": "we have", "what'll": "what will",
                     "what're": "what are", "what's": "what is", "what've": "what have", "where's": "where is",
                     "who'd": "who would", "who'll": "who will", "who're": "who are", "who's": "who is",
                     "who've": "who have", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                     "you'll": "you will", "you're": "you are", "you've": "you have", "'re": "are", "wasn't": "was not",
                     "we'll":"will", "tryin'":"trying", "arent": "are not", "cant": "cannot", "couldnt": "could not",
                     "didnt": "did not", "doesnt": "does not", "dont": "do not", "hadnt": "had not", "hasnt": "has not",
                     "havent": "have not", "isnt": "is not", "its": "it is", "itll":"it will", "ive": "I have",
                     "lets": "let us", "mightnt": "might not", "mustnt": "must not", "shant": "shall not",
                     "shed": "she would", "shell": "she will", "shes": "she is", "shouldnt": "should not",
                     "thats": "that is", "theres": "there is", "theyd": "they would", "theyll": "they will",
                     "theyre": "they are", "theyve": "they have", "wed": "we would", "were": "we are",
                     "werent": "were not", "weve": "we have", "whatll": "what will", "whatre": "what are",
                     "whats": "what is", "whatve": "what have", "wheres": "where is", "whod": "who would",
                     "wholl": "who will", "whore": "who are", "whos": "who is", "whove": "who have", "wont": "will not",
                     "wouldnt": "would not", "youd": "you would", "youll": "you will", "youre": "you are",
                     "youve": "you have", "wasnt": "was not"}


def remove_duplicates(reviews):
    """
    Remove duplicate records from the dataset

    Args:
        reviews(pd.DataFrame): The dataset

    Returns:
       pd.DataFrame. The dataset without duplicate records
    """
    features_subset = reviews.columns[1:]  # Remove index
    return reviews.drop_duplicates(features_subset, keep='first')


def calculate_helpfulness_rank(reviews):
    """
    Calculate the helpfulness rank, the ratio of users who ranked the review as helpful

    Args:
        reviews(pd.DataFrame): The dataset

    Returns:
        pd.DataFrame. The dataset including the helpfulness rank as a column
    """
    reviews['HelpfulnessRank'] = round(reviews.HelpfulnessNumerator / reviews.HelpfulnessDenominator, 1) * 10
    return reviews


def remove_nans(reviews):
    """
    Remove nan values from the dataset

    Args:
        reviews(pd.DataFrame): the dataset

    Returns:
        pd.DataFrame. the dataset without nan values
    """
    reviews = reviews[~reviews.HelpfulnessRank.isnull()]
    return reviews.dropna()


def down_sample_helpfulness_rank_randomly(reviews, percentage, helpfulness_rank):
    """
    Down sample randomly percentage of the records with the specified helpfulness_rank

    Args:
        reviews(pd.DataFrame): the dataset
        percentage(float): Precentage that we down sample accordingly
        helpfulness_rank(int): The helpfulness rank that we would like to downsample

    Returns:
        pd.DataFrame. the dataset post down sampling a percentage of the helpfulness rank records
    """
    reviews_by_rank = reviews[reviews.HelpfulnessRank == helpfulness_rank]
    mask_reviews_by_rank = reviews[reviews.HelpfulnessRank != helpfulness_rank]
    down_sample_reviews_by_rank = reviews_by_rank.sample(frac=percentage)
    return pd.concat([mask_reviews_by_rank, down_sample_reviews_by_rank])


def get_user_reviews_count(reviews):
    """
    Count the number of reviews writen by each user and adds it as a 'user_reviews_count' column

    Args:
        reviews(pd.DataFrame): the dataset

    Returns:
        pd.DataFrame. with the 'user_reviews_count' column
    """
    users_reviews_count = reviews.ProfileName.value_counts()
    users_reviews_count = users_reviews_count.to_frame(name='user_reviews_count').reset_index(
                                                       ).rename(columns={'index': 'ProfileName'})
    return reviews.merge(users_reviews_count)


def get_product_frequency(reviews):
    """
    Count the occurrence of each product and adds it as a products frequency column

    Args:
        reviews(pd.DataFrame): the dataset

    Returns:
        pd.DataFrame. the dataset with a product frequency column
    """
    product_occurrence = pd.DataFrame(reviews.ProductId.value_counts()).reset_index()
    product_occurrence = product_occurrence.rename(columns={"index": "ProductId", "ProductId": "freq"})
    reviews = reviews.merge(product_occurrence)
    return reviews.rename(columns={'freq': 'ProductFrequency'})


def remove_rare_products(reviews, min_reviews):
    """
    Remove product whose occurrence is smaller than min_reviews

    Args:
        reviews(pd.DataFrame): the dataset
        min_review(int): the minimum reviews per product threshold

    Returns
        pd.DataFrame: the dataset with of rare product filtered
    """
    return reviews[reviews.ProductFrequency >= min_reviews]


def calculate_helpfulness_rank_variance(reviews):
    """
    Calculate the variance of the helpfulness rank among the product's reviews

    Args:
        reviews(pd.DataFrame): the dataset

    Returns:
        pd.DataFrame. the dataset with helpfulness_var column
    """
    helpfulness_variance = reviews.groupby('ProductId').agg('var')['HelpfulnessRank']
    rank_variance = pd.DataFrame(
        {'ProductId': np.array(helpfulness_variance.index), 'HelpfulnessVar': helpfulness_variance.values})
    return reviews.merge(rank_variance)


def clean_text(sentence):
    """
    Clean text  - tokenize, abbreviation, uppercase and etc

    Args:
        sentence(str): a sentence text

    Returns:
        str. a clean sentence text
    """
    parser = English()
    parser.vocab["not"].is_stop = False
    parser.vocab["cannot"].is_stop = False
    sentence = sentence.lower()  # lower case Hi -> hi
    sentence = re.sub("\\n", " ", sentence)  # remove \n

    tokenizer = TweetTokenizer()
    words = tokenizer.tokenize(sentence)
    words = ' '.join([APOSTROPHE_LOOKUP[word] if word in APOSTROPHE_LOOKUP else word for word in words])
    words = [token.lemma_ for token in parser(words) if (not token.is_stop and not token.is_space)]
    words = [word for word in words if word not in ['"', 's', '.', ';', ',']]
    # words = [w for w in words if not w in eng_stopwords]
    clean_sentence = " ".join(words)
    return clean_sentence


def pre_process_dataset(reviews):
    """
    Pre-Process the dataset and save it into interim file

    Args:
        reviews(pd.DataFrame): the dataset

    Returns
        pd.DataFrame. the dataset post pre-processing
    """
    pre_process_functions = [remove_duplicates, calculate_helpfulness_rank, calculate_helpfulness_rank_variance,
                             remove_nans, down_sample_helpfulness_rank_randomly, get_user_reviews_count,
                             get_product_frequency, remove_rare_products]

    for func in pre_process_functions:
        reviews = func(reviews)
        if func == down_sample_helpfulness_rank_randomly:
            reviews = func(reviews, DOWN_SAMPLE_RATIO, HELPFULNESS_RANK)

    reviews = reviews.Text.agg(clean_text)
    with open(os.path.join('data', 'interim', 'reviews_post_processing.pkl'), 'wb') as f:
        pickle.dump(reviews, f)

    return reviews
