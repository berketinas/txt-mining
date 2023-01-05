import numpy as np
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

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(tfidf)

    # accuracy = kmeans.score(tfidf)
    # print(f'K-MEANS ACCURACY: {accuracy}')

    reviews['predicted_by_kmeans'] = kmeans.predict(tfidf)
    return reviews


def linear_regression(reviews, tfidf):
    x_train, x_test, y_train, y_test = train_test_split(tfidf,
                                                        reviews.sentiment,
                                                        test_size=0.3,
                                                        random_state=77)
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train.astype('int').values)
    accuracy = lin_reg.score(x_test, y_test.astype('int').values)
    print(f'LINEAR REGRESSION ACCURACY: {accuracy}')


def naive_bayes(reviews, tfidf):
    x_train, x_test, y_train, y_test = train_test_split(tfidf,
                                                        reviews.sentiment,
                                                        test_size=0.3,
                                                        random_state=77)
    nb = GaussianNB()
    nb.fit(x_train, y_train.astype('int').values)
    accuracy = nb.score(x_test, y_test.astype('int').values)
    print(f'NAIVE BAYES ACCURACY: {accuracy}')


def decision_tree(reviews, tfidf):
    x_train, x_test, y_train, y_test = train_test_split(tfidf,
                                                        reviews.sentiment,
                                                        test_size=0.3,
                                                        random_state=77)

    dtc = DecisionTreeClassifier()
    dtc = dtc.fit(x_train, y_train.astype('int').values)
    accuracy = dtc.score(x_test, y_test.astype('int').values)
    print(f'DECISION TREE ACCURACY: {accuracy}')


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
