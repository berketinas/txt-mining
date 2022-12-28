import os
import string
import re

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from sklearn.linear_model import LinearRegression

from textblob import TextBlob

from wordcloud import WordCloud

from IPython.display import display

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

POS_DATA_PATH = './dataset/pos'
NEG_DATA_PATH = './dataset/neg'

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
stop_words = stop_words + list(string.printable)

intensity_words = set(wordnet.words())

reviews_clean_df = pd.DataFrame(columns=['cleaned_text', 'sentiment'])
tfidf_vectors = pd.DataFrame()

MAX_FEATURES = 10


def read(file_path, sentiment):
    with open(file_path, 'r') as read_pos:
        return preprocess(read_pos.read(), sentiment)


def preprocess(review, sentiment):
    reviews_text_df = pd.DataFrame({'text': [review]})
    temp_clean_df = pd.DataFrame(columns=['cleaned_text', 'sentiment'])

    temp_clean_df['cleaned_text'] = reviews_text_df['text'].apply(lambda x: ' '.join(
        [lemmatizer.lemmatize(word) for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x))) if
         word not in stop_words]))
    temp_clean_df['sentiment'] = sentiment

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
    reviews['intensity'] = reviews['cleaned_text']\
        .apply(lambda x: str(TextBlob(x).subjectivity))

    tfidf_model = TfidfVectorizer(max_features=250)
    tfidf = pd.DataFrame(tfidf_model.fit_transform(reviews['cleaned_text']).todense())
    tfidf.columns = sorted(tfidf_model.vocabulary_)

    return reviews, tfidf


def polarity_test(reviews):
    sum_pos_polarity = 0
    sum_neg_polarity = 0
    sum_false = 0

    for iterator in range(2000):
        if float(reviews.iloc[iterator, 2]) > 0.09 and iterator <= 1000:
            sum_pos_polarity += 1

        elif float(reviews.iloc[iterator, 2]) < 0.09 and iterator > 1000:
            sum_neg_polarity += 1

        else:
            sum_false += 1

    print('sum_pos_polarity: ', sum_pos_polarity)
    print('sum_neg_polarity: ', sum_neg_polarity)
    print('sum_false: ', sum_false)


def linreg_test(reviews):
    sum_pos_linreg = 0
    sum_neg_linreg = 0
    sum_false = 0

    # 0.07 646
    # 0.06 654
    # 0.09 617
    # 0.1 613

    for iterator in range(2000):
        if float(reviews.iloc[iterator, 5]) > 0.09 and iterator <= 1000:
            sum_pos_linreg += 1

        elif float(reviews.iloc[iterator, 5]) < 0.09 and iterator > 1000:
            sum_neg_linreg += 1

        else:
            sum_false += 1

    print('sum_pos_linreg: ', sum_pos_linreg)
    print('sum_neg_linreg: ', sum_neg_linreg)
    print('sum_false: ', sum_false)


def word_cloud(reviews):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          max_words=50,
                          min_font_size=10).generate(str(reviews['cleaned_text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def k_means_clustering(tfidf):
    distortions = []
    K = range(1, 6)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tfidf)
        distortions.append(sum(np.min(cdist(tfidf, kmeans.cluster_centers_, 'euclidean'), axis=1)) / tfidf.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('distortion')
    plt.title('elbow method for optimal k')
    plt.show()

    # kmeans = KMeans(n_clusters=OPTIMAL_K)
    # kmeans.fit(tfidf)


def linear_regression(reviews, tfidf):
    lin_reg = LinearRegression()
    lin_reg.fit(tfidf, reviews['polarity'])
    reviews['predicted_by_linear_regression'] = lin_reg.predict(tfidf)
    display(reviews)


def logistic_regression(reviews, tfidf):

    logreg = LogisticRegression()
    logreg.fit(tfidf, reviews['sentiment'])
    predicted_labels = logreg.predict(tfidf)

    reviews['predicted_labels'] = predicted_labels
    return reviews

def naive_bayes(reviews,tfidf):

    nb = GaussianNB()
    nb.fit(tfidf, reviews['sentiment'])
    predicted_labels = nb.predict(tfidf)
    reviews['predicted_labels_nb'] = predicted_labels
    return reviews





for file in os.listdir(POS_DATA_PATH):
    FILE_PATH = f"{POS_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH, 'pos')], axis=0, ignore_index=True)

for file in os.listdir(NEG_DATA_PATH):
    FILE_PATH = f"{NEG_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH, 'neg')], axis=0, ignore_index=True)

reviews_clean_df, tfidf_vectors = feature_extraction(reviews_clean_df)

reviews_clean_df['sentiment'] = reviews_clean_df['sentiment'].apply(lambda x : 0 if x =='neg'
else 1)

reviews_clean_df = logistic_regression(reviews_clean_df,tfidf_vectors)
reviews_clean_df = naive_bayes(reviews_clean_df,tfidf_vectors)

# display(pd.crosstab(reviews_clean_df['sentiment'], reviews_clean_df['predicted_labels']))
# display(pd.crosstab(reviews_clean_df['sentiment'], reviews_clean_df['predicted_labels_nb']))


