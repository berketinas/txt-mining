import os
import string
import re

import pandas as pd

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

pd.set_option('display.max_colwidth', 200)

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

print(reviews_clean_df)
