import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from textblob import TextBlob
from wordcloud import WordCloud
from IPython.display import display
from sklearn.metrics import roc_curve, auc, mean_squared_error
from math import sqrt

MAX_FEATURES = 10
    #part B.1
def feature_extraction(reviews):
    reviews['polarity'] = reviews['cleaned_text'].apply(lambda x: str(TextBlob(x).sentiment.polarity))
    reviews['word_count'] = reviews['cleaned_text'].apply(lambda x: str(len(x.split())))
    reviews['subjectivity'] = reviews['cleaned_text']\
        .apply(lambda x: str(TextBlob(x).subjectivity))

    tfidf_model = TfidfVectorizer()
    tfidf = pd.DataFrame(tfidf_model.fit_transform(reviews['cleaned_text']).todense())
    tfidf.columns = sorted(tfidf_model.vocabulary_)

    return reviews, tfidf

    # part B.2
def b2models(reviews):
    bow_model = CountVectorizer(max_features=MAX_FEATURES)
    bow_df = pd.DataFrame(bow_model.fit_transform(reviews['cleaned_text']).todense())
    bow_df.columns = sorted(bow_model.vocabulary_)

    display(bow_df)

    tfidf_model = TfidfVectorizer(max_features=MAX_FEATURES)
    tfidf_df = pd.DataFrame(tfidf_model.fit_transform(reviews['cleaned_text']).todense())
    tfidf_df.columns = sorted(tfidf_model.vocabulary_)

    display(tfidf_df)

    #part B.3
def word_cloud(reviews):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          max_words=50,
                          min_font_size=10).generate(str(reviews['cleaned_text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


    #part B.4
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
