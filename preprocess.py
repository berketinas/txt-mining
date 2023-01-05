import string
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
stop_words = stop_words + list(string.printable)


def read(file_path, sentiment):
    with open(file_path, 'r') as read_pos:
        return preprocess(read_pos.read(), sentiment)


def preprocess(review, sentiment):
    reviews_text_df = pd.DataFrame({'text': [review]})
    temp_clean_df = pd.DataFrame()

    # count exclamation marks here, because punctuation is removed
    # during preprocessing
    temp_clean_df['ex_mark'] = reviews_text_df['text'].apply(lambda x: x.count('!'))

    # lemmatize, remove stop words,
    temp_clean_df['cleaned_text'] = reviews_text_df['text'].apply(lambda x: ' '.join(
        [lemmatizer.lemmatize(word) for word in word_tokenize(re.sub(r'([^\s\w]|_)+', ' ', str(x))) if
         word not in stop_words]))

    temp_clean_df['sentiment'] = reviews_text_df['text'].apply(lambda x: int(sentiment))

    return temp_clean_df
