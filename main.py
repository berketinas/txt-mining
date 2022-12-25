import os
import string
import re

import pandas as pd

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

from IPython.display import display

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

POS_DATA_PATH = './dataset/pos'
NEG_DATA_PATH = './dataset/neg'

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
stop_words = stop_words + list(string.printable)

reviews_clean_df = pd.DataFrame(columns=['cleaned_text'])

MAX_FEATURES = 10


def read(file_path):
    with open(file_path, 'r') as read_pos:
        return preprocess(read_pos.read())


def preprocess(review):
    reviews_text_df = pd.DataFrame({'text': [review]})
    temp_clean_df = pd.DataFrame(columns=['cleaned_text'])

    temp_clean_df['cleaned_text'] = reviews_text_df['text'].apply(lambda x: ' '.join(
        [lemmatizer.lemmatize(word) for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x))) if
         word not in stop_words]))

    return temp_clean_df


def b2models(reviews):
    bow_model = CountVectorizer(max_features=MAX_FEATURES)
    bow_df = pd.DataFrame(bow_model.fit_transform(reviews['cleaned_text']).todense())
    bow_df.columns = sorted(bow_model.vocabulary_)

    display(bow_df)

    tfidf_model = TfidfVectorizer(max_features=MAX_FEATURES)
    tfidf_df = pd.DataFrame(tfidf_model.fit_transform(reviews['cleaned_text']).todense())
    tfidf_df.columns = sorted(tfidf_model.vocabulary_)

    display(tfidf_df)


def feature_extraction(reviews):
    reviews['polarity'] = reviews['cleaned_text'].apply(lambda x: str(TextBlob(x).sentiment.polarity))
    reviews['word_count'] = reviews['cleaned_text'].apply(lambda x: str(len(x.split())))

    display(reviews)
    return reviews


def polarity_test(reviews):
    sum_pos_polarity = 0
    sum_neg_polarity = 0
    sum_false = 0

    for iterator in range(2000):
        if float(reviews.iloc[iterator, 1]) > 0.09 and iterator <= 1000:
            sum_pos_polarity += 1

        elif float(reviews.iloc[iterator, 1]) < 0.09 and iterator > 1000:
            sum_neg_polarity += 1

        else:
            sum_false += 1

    print('sum_pos_polarity: ', sum_pos_polarity)
    print('sum_neg_polarity: ', sum_neg_polarity)
    print('sum_false: ', sum_false)


for file in os.listdir(POS_DATA_PATH):
    FILE_PATH = f"{POS_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH)], axis=0, ignore_index=True)

for file in os.listdir(NEG_DATA_PATH):
    FILE_PATH = f"{NEG_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH)], axis=0, ignore_index=True)

reviews_clean_df = feature_extraction(reviews_clean_df)

