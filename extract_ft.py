import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud
from IPython.display import display


MAX_FEATURES = 10


def feature_extraction(reviews):
    # three general features polarity, word count,
    # and subjectivity are extracted
    reviews['polarity'] = reviews['cleaned_text'].apply(lambda x: str(TextBlob(x).sentiment.polarity))
    reviews['word_count'] = reviews['cleaned_text'].apply(lambda x: str(len(x.split())))
    reviews['subjectivity'] = reviews['cleaned_text']\
        .apply(lambda x: str(TextBlob(x).subjectivity))

    # tf-idf is extracted as the specific feature
    tfidf_model = TfidfVectorizer()
    tfidf = pd.DataFrame(tfidf_model.fit_transform(reviews['cleaned_text']).todense())
    tfidf.columns = sorted(tfidf_model.vocabulary_)

    return reviews, tfidf


def compare_bow_tfidf(reviews):
    bow_model = CountVectorizer(max_features=MAX_FEATURES)
    bow_df = pd.DataFrame(bow_model.fit_transform(reviews['cleaned_text']).todense())
    bow_df.columns = sorted(bow_model.vocabulary_)

    display(bow_df)

    tfidf_model = TfidfVectorizer(max_features=MAX_FEATURES)
    tfidf_df = pd.DataFrame(tfidf_model.fit_transform(reviews['cleaned_text']).todense())
    tfidf_df.columns = sorted(tfidf_model.vocabulary_)

    display(tfidf_df)


def word_cloud(reviews):
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          max_words=50,
                          min_font_size=10).generate(str(reviews['cleaned_text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
