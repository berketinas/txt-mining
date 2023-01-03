import os
import string
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc, mean_squared_error

from textblob import TextBlob

from wordcloud import WordCloud

from IPython.display import display

from math import sqrt


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

MAX_FEATURES = 10


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


def split_data(reviews):
    return train_test_split(reviews.cleaned_text, reviews.sentiment, test_size=0.3, random_state=77)


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
    reviews['subjectivity'] = reviews['cleaned_text']\
        .apply(lambda x: str(TextBlob(x).subjectivity))

    tfidf_model = TfidfVectorizer()
    tfidf = pd.DataFrame(tfidf_model.fit_transform(reviews['cleaned_text']).todense())
    tfidf.columns = sorted(tfidf_model.vocabulary_)

    return reviews, tfidf


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


def word_cloud(reviews):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          max_words=50,
                          min_font_size=10).generate(str(reviews['cleaned_text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def k_means_clustering(reviews, tfidf):
    distortions = []
    K = range(1, 40)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tfidf)
        distortions.append(sum(np.min(cdist(tfidf, kmeans.cluster_centers_, 'euclidean'), axis=1)) / tfidf.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('distortion')
    plt.title('elbow method for optimal k')
    plt.show()

    # kmeans = KMeans(n_clusters=3)
    # kmeans.fit(tfidf)
    # reviews['predicted_by_kmeans'] = kmeans.predict(tfidf)
    #
    # return reviews


def linear_regression(reviews, tfidf):
    lin_reg = LinearRegression()
    lin_reg.fit(tfidf, reviews['polarity'])
    reviews['predicted_by_linear_regression'] = lin_reg.predict(tfidf)

    return reviews


def logistic_regression(reviews, tfidf):
    logreg = LogisticRegression()
    logreg.fit(tfidf, reviews['sentiment'])
    predicted_labels = logreg.predict(tfidf)

    reviews['predicted_labels'] = predicted_labels
    return reviews


def naive_bayes(reviews, tfidf):
    nb = GaussianNB()
    nb.fit(tfidf, reviews['sentiment'])
    predicted_labels = nb.predict(tfidf)
    reviews['predicted_labels_nb'] = predicted_labels
    return reviews


def decision_tree(reviews, tfidf):
    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(tfidf, reviews['sentiment'])
    reviews['predicted_by_dtc'] = dtc.predict(tfidf)

    return reviews


def part_e(reviews):
    x_train, x_test, y_train, y_test = train_test_split(reviews.cleaned_text, reviews.sentiment, test_size=0.3, random_state=42)

    log_reg = LogisticRegression(random_state=0, solver='lbfgs')
    tfidf = TfidfVectorizer(strip_accents=None, preprocessor=None, lowercase=False)
    log_tfidf = Pipeline([('vect', tfidf), ('clf', log_reg)])

    y_train = y_train.astype('int')
    log_tfidf.fit(x_train, y_train.values)

    y_test = y_test.astype('int')
    accuracy = log_tfidf.score(x_test, y_test.values)
    print(f'accuracy: {accuracy}')


# Precision = True Positive / (True Positive + False Positive)
# Recall = True Positive / (True Positive + False Negative)
def precision_recall(true_pos, false_pos, false_neg):
    return true_pos / (true_pos + false_pos), true_pos / (true_pos + false_neg)


def f1_score(true_pos, false_pos, false_neg):
    precision, recall = precision_recall(true_pos, false_pos, false_neg)
    return 2 * ((precision * recall) / (precision + recall))


def confusion_matrix(actual, predicted):
    display(pd.crosstab(actual, predicted))


def roc(actual, predicted):
    fpr, tpr, threshold = roc_curve(actual, predicted)

    print('Area under ROC curve for validation set: ', auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label='Validation set AUC')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    ax.legend(loc='best')
    plt.show()


def rmse(actual, predicted):
    print('ROOT MEAN SQUARE ERROR: ', sqrt(mean_squared_error(actual, predicted)))


for file in os.listdir(POS_DATA_PATH):
    FILE_PATH = f"{POS_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH, 1)], axis=0, ignore_index=True)

for file in os.listdir(NEG_DATA_PATH):
    FILE_PATH = f"{NEG_DATA_PATH}\{file}"
    reviews_clean_df = pd.concat([reviews_clean_df, read(FILE_PATH, 0)], axis=0, ignore_index=True)

reviews_clean_df, tfidf_vectors = feature_extraction(reviews_clean_df)

part_e(reviews_clean_df)

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