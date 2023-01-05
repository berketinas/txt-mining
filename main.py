import os
import pandas as pd
from IPython.core.display_functions import display

from preprocess import read
from extract_ft import feature_extraction, compare_bow_tfidf, word_cloud
from algorithms import k_means_clustering, linear_regression, naive_bayes, decision_tree, logistic_regression
from perf_measure import precision_recall, f1_score, confusion_matrix, roc, rmse

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

POS_DATA_PATH = './dataset/pos'
NEG_DATA_PATH = './dataset/neg'

reviews = pd.DataFrame(columns=['cleaned_text', 'sentiment'])
tfidf = pd.DataFrame()

# read the reviews from specified directories
# read() function from preprocess.py also preprocesses the text
for file in os.listdir(POS_DATA_PATH):
    FILE_PATH = f"{POS_DATA_PATH}\{file}"
    reviews = pd.concat([reviews, read(FILE_PATH, 1)], axis=0, ignore_index=True)

for file in os.listdir(NEG_DATA_PATH):
    FILE_PATH = f"{NEG_DATA_PATH}\{file}"
    reviews = pd.concat([reviews, read(FILE_PATH, 0)], axis=0, ignore_index=True)

# extract general and specific features
reviews, tfidf = feature_extraction(reviews)

# compare bag of words and tfidf top 10 words
# compare_bow_tfidf(reviews)

# word cloud of top 50 words
# word_cloud(reviews)

# k-means clustering
# reviews = k_means_clustering(reviews, tfidf)
# linear regression
linear_regression(reviews, tfidf)

# naive bayes
naive_bayes(reviews, tfidf)

# decision tree
decision_tree(reviews, tfidf)

# logistic regression
logistic_regression(reviews)
