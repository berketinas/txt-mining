import os
import string
import re

import pandas as pd

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

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


for file in os.listdir(POS_DATA_PATH):
    FILE_PATH = f"{POS_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH)], axis=0, ignore_index=True)

for file in os.listdir(NEG_DATA_PATH):
    FILE_PATH = f"{NEG_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH)], axis=0, ignore_index=True)

bow_model = CountVectorizer(max_features=10)
bow_df = pd.DataFrame(bow_model.fit_transform(reviews_clean_df['cleaned_text']).todense())
bow_df.columns = sorted(bow_model.vocabulary_)

display(bow_df)

tfidf_model = TfidfVectorizer(max_features=10)
tfidf_df = pd.DataFrame(tfidf_model.fit_transform(reviews_clean_df['cleaned_text']).todense())
tfidf_df.columns = sorted(tfidf_model.vocabulary_)

display(tfidf_df)

# rw = 8
# print(list(bow_df.columns[bow_df.iloc[rw, :] == bow_df.iloc[rw, :].max()]))
# print(list(tfidf_df.columns[tfidf_df.iloc[rw, :] == tfidf_df.iloc[rw, :].max()]))
#
# print(bow_df[bow_df['life']!=0].shape[0])
# print(bow_df[bow_df['like']!=0].shape[0])
