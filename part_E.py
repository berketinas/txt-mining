from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def split_data(reviews):
    return train_test_split(reviews.cleaned_text, reviews.sentiment, test_size=0.3, random_state=77)


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


