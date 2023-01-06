import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from perf_measure import confusion_matrix, roc, rmse, f1_score


def k_means_clustering(reviews, tfidf):
    # graph to find optimal k
    # distortions = []
    # K = range(1, 8)
    # for k in K:
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(tfidf)
    #     distortions.append(sum(np.min(cdist(tfidf, kmeans.cluster_centers_, 'euclidean'), axis=1)) / tfidf.shape[0])
    #
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('distortion')
    # plt.title('elbow method for optimal k')
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(tfidf,
                                                        reviews.sentiment,
                                                        test_size=0.3,
                                                        random_state=77)

    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(x_train)

    predicted = kmeans.predict(x_test)
    confusion_matrix(y_test.astype('int'), predicted)
    roc(y_test.astype('int'), predicted)
    rmse(y_test.astype('int'), predicted)


def linear_regression(reviews, tfidf):
    x_train, x_test, y_train, y_test = train_test_split(tfidf,
                                                        reviews.sentiment,
                                                        test_size=0.3,
                                                        random_state=77)
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train.astype('int').values)
    accuracy = lin_reg.score(x_test, y_test.astype('int').values)
    print(f'LINEAR REGRESSION ACCURACY: {accuracy}')

    predicted = lin_reg.predict(x_test).astype('int')

    true_pos, false_pos, false_neg = 0, 0, 0
    for i in range(0, 600):
        if y_test.astype('int').values[i] == 1:
            if predicted[i] == 1:
                true_pos += 1

            else:
                false_neg += 1

        else:
            if predicted[i] == 1:
                false_pos += 1

    f1_score(true_pos, false_pos, false_neg)
    roc(y_test.astype('int'), predicted)
    rmse(y_test.astype('int'), predicted)


def naive_bayes(reviews, tfidf):
    x_train, x_test, y_train, y_test = train_test_split(tfidf,
                                                        reviews.sentiment,
                                                        test_size=0.3,
                                                        random_state=77)
    nb = GaussianNB()
    nb.fit(x_train, y_train.astype('int').values)
    accuracy = nb.score(x_test, y_test.astype('int').values)
    print(f'NAIVE BAYES ACCURACY: {accuracy}')

    predicted = nb.predict(x_test)

    true_pos, false_pos, false_neg = 0, 0, 0
    for i in range(0, 600):
        if y_test.astype('int').values[i] == 1:
            if predicted[i] == 1:
                true_pos += 1

            else:
                false_neg += 1

        else:
            if predicted[i] == 1:
                false_pos += 1

    f1_score(true_pos, false_pos, false_neg)
    confusion_matrix(y_test.astype('int'), predicted)
    roc(y_test.astype('int'), predicted)
    rmse(y_test.astype('int'), predicted)


def decision_tree(reviews, tfidf):
    x_train, x_test, y_train, y_test = train_test_split(tfidf,
                                                        reviews.sentiment,
                                                        test_size=0.5,
                                                        random_state=77)

    dtc = DecisionTreeClassifier(max_depth=5)
    dtc = dtc.fit(x_train, y_train.astype('int').values)
    accuracy = dtc.score(x_test, y_test.astype('int').values)
    print(f'DECISION TREE ACCURACY: {accuracy}')

    predicted = dtc.predict(x_test)

    true_pos, false_pos, false_neg = 0, 0, 0
    for i in range(0, 600):
        if y_test.astype('int').values[i] == 1:
            if predicted[i] == 1:
                true_pos += 1

            else:
                false_neg += 1

        else:
            if predicted[i] == 1:
                false_pos += 1

    f1_score(true_pos, false_pos, false_neg)
    confusion_matrix(y_test.astype('int'), predicted)
    roc(y_test.astype('int'), predicted)
    rmse(y_test.astype('int'), predicted)


# FOR PART E
def logistic_regression(reviews):
    # randomly split the data, 70% train set - 30% train set
    x_train, x_test, y_train, y_test = train_test_split(reviews.cleaned_text,
                                                        reviews.sentiment,
                                                        test_size=0.3,
                                                        random_state=77)

    # initialize the logistic regressor and tf-idf vectorizer
    # then plug them into the pipeline
    log_reg = LogisticRegression(random_state=0, solver='lbfgs')
    tfidf = TfidfVectorizer(strip_accents=None, preprocessor=None, lowercase=False)
    log_tfidf = Pipeline([('vect', tfidf), ('clf', log_reg)])

    # train and predict with corresponding sets, calculate accuracy
    y_train = y_train.astype('int')
    log_tfidf.fit(x_train, y_train.values)

    y_test = y_test.astype('int')
    accuracy = log_tfidf.score(x_test, y_test.values)
    print(f'LOGISTIC REGRESSION ACCURACY: {accuracy}')

    predicted = log_tfidf.predict(x_test)

    true_pos, false_pos, false_neg = 0, 0, 0
    for i in range(0, 600):
        if y_test.astype('int').values[i] == 1:
            if predicted[i] == 1:
                true_pos += 1

            else:
                false_neg += 1

        else:
            if predicted[i] == 1:
                false_pos += 1

    f1_score(true_pos, false_pos, false_neg)
    confusion_matrix(y_test, predicted)
    roc(y_test, predicted)
    rmse(y_test, predicted)
