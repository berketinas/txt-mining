import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from main import reviews_clean_df, display
from nltk import sent_tokenize, word_tokenize
import contractions

GLOVE_DIR = 'dataset/glove/'
GLOVE_FILE = 'glove.6B.50d.txt'
GLOVE_PATH = GLOVE_DIR + GLOVE_FILE

VECTOR_SIZE = 50
EMPTY_VECTOR = np.zeros(VECTOR_SIZE)


def load_glove_vectors(fn):
    print("Loading Glove Model")
    with open(fn, 'r', encoding='utf8') as glove_vector_file:
        model = {}
        for line in glove_vector_file:
            parts = line.split()
            word = parts[0]
            embedding = np.array([float(val) for val in parts[1:]])
            model[word] = embedding
        print("Loaded {} words".format(len(model)))
    return model


# create the sentence vectors
def sentence_vector(sentence):
    return sum([glove_vectors.get(word, EMPTY_VECTOR) for word in sentence]) / len(sentence)


def sentences_to_vectors(sentences):
    return [sentence_vector(sentence) for sentence in sentences]


# captures the degree to which one sentence is similar to another.

def similarity_matrix(sentence_vectors):
    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):
            element_i = sentence_vectors[i].reshape(1, VECTOR_SIZE)
            element_j = sentence_vectors[j].reshape(1, VECTOR_SIZE)
            sim_mat[i][j] = cosine_similarity(element_i, element_j)[0, 0]
    return sim_mat


def compute_graph(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    return scores


def tokenize_words(sentences):
    return [word_tokenize(sentence) for sentence in sentences]


# The graph contains a score and a numeric index to a sentence. We have to write
# a function that will rank the scores and return the top n sentences by their graph
# scores
def get_ranked_sentences(sentences, scores, n=3):
    top_scores = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n_sentences = [sentence for score, sentence in top_scores[:n]]
    return " ".join(top_n_sentences)


def fix_contractions(sentences):
    return [contractions.fix(sentence) for sentence in sentences]


# sentences in article = reviews.cleaned_text.apply(sent_tokenize)
reviews_clean_df['sentences_in_review'] = reviews_clean_df.cleaned_text.apply(sent_tokenize)

# words in sentences = reviews.sentences_in_review.apply(tokenize_words)
reviews_clean_df['words_in_sentences'] = reviews_clean_df.sentences_in_review.apply(tokenize_words)

glove_vectors = load_glove_vectors(GLOVE_PATH)
reviews_clean_df['sentence_vector'] = reviews_clean_df.words_in_sentences.apply(sentences_to_vectors)
reviews_clean_df['sim_matrix'] = reviews_clean_df.sentence_vector.apply(similarity_matrix)
reviews_clean_df['graph'] = reviews_clean_df.sim_matrix.apply(compute_graph)
reviews_clean_df['summary'] = reviews_clean_df \
    .apply(lambda d: get_ranked_sentences(d.sentences_in_review, d.graph), axis=1)
