import os
import string
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

POS_DATA_PATH = './dataset/pos'
NEG_DATA_PATH = './dataset/neg'

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
stop_words = stop_words + list(string.printable)

reviews_clean_df = pd.DataFrame(columns=['cleaned_text', 'sentiment'])
tfidf_vectors = pd.DataFrame()


def read(file_path, sentiment):
    with open(file_path, 'r') as read_pos:
        return preprocess(read_pos.read(), sentiment)


def preprocess(review, sentiment):
    reviews_text_df = pd.DataFrame({'text': [review]})
    temp_clean_df = pd.DataFrame(columns=['cleaned_text', 'sentiment'])

    temp_clean_df['ex_mark'] = reviews_text_df['text'].apply(lambda x: x.count('!'))
    temp_clean_df['cleaned_text'] = reviews_text_df['text'].apply(lambda x: ' '.join(
        [lemmatizer.lemmatize(word) for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x))) if
         word not in stop_words]))
    temp_clean_df['sentiment'] = reviews_text_df['text'].apply(lambda x: sentiment)

    return temp_clean_df

def test(reviews):
    sum_pos = 0
    sum_neg = 0
    sum_false = 0
    # 0.07 646
    # 0.06 654
    # 0.09 617
    # 0.1 613
    for iterator in range(2000):
        if float(reviews.iloc[iterator, 5]) > 0.09 and iterator <= 1000:
            sum_pos += 1
        elif float(reviews.iloc[iterator, 5]) < 0.09 and iterator > 1000:
            sum_neg += 1
        else:
            sum_false += 1
    print('sum_pos: ', sum_pos)
    print('sum_neg: ', sum_neg)
    print('sum_false: ', sum_false)



for file in os.listdir(POS_DATA_PATH):
    FILE_PATH = f"{POS_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH, 1)], axis=0, ignore_index=True)

for file in os.listdir(NEG_DATA_PATH):
    FILE_PATH = f"{NEG_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH, 0)], axis=0, ignore_index=True)

#reviews_clean_df, tfidf_vectors = feature_extraction(reviews_clean_df)

#part_e(reviews_clean_df)

# reviews_clean_df = decision_tree(reviews_clean_df, tfidf_vectors)
#
# reviews_clean_df = logistic_regression(reviews_clean_df, tfidf_vectors)
#
# reviews_clean_df = naive_bayes(reviews_clean_df, tfidf_vectors)
#
# reviews_clean_df = linear_regression(reviews_clean_df, tfidf_vectors)
#
# reviews_clean_df = k_means_clustering(reviews_clean_df, tfidf_vectors)

# test(reviews_clean_df)
# rmse(reviews_clean_df['sentiment'], reviews_clean_df['predicted_labels'])