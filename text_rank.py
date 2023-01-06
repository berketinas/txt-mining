import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk import sent_tokenize, word_tokenize


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


def sentence_vector(sentence):
    return sum([glove_vectors.get(word, EMPTY_VECTOR) for word in sentence]) / len(sentence)


# create the sentence vectors
def sentences_to_vectors(sentences):
    return [sentence_vector(sentence) for sentence in sentences]


# calculate the degree to which one sentence is similar to another
def similarity_matrix(sentence_vectors):
    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):
            element_i = sentence_vectors[i].reshape(1, VECTOR_SIZE)
            element_j = sentence_vectors[j].reshape(1, VECTOR_SIZE)
            sim_mat[i][j] = cosine_similarity(element_i, element_j)[0, 0]
    return sim_mat


# create the graph which will help determine relative importance
def compute_graph(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    return scores


def tokenize_words(sentences):
    return [word_tokenize(sentence) for sentence in sentences]


# calculate scores of sentences by their graph scores
# return top n sentences
def get_ranked_sentences(sentences, scores, n=3):
    top_scores = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n_sentences = [sentence for score, sentence in top_scores[:n]]
    return " ".join(top_n_sentences)


def write_file(df, sentiment):
    with open('{}_summarized.txt'.format(sentiment), 'w') as f:
        for review in df:
            f.write(review)
            f.write('\n')


def text_rank(reviews):
    reviews['sentences_in_review'] = reviews.cleaned_text.apply(sent_tokenize)
    reviews['words_in_sentences'] = reviews.sentences_in_review.apply(tokenize_words)

    reviews['sentence_vector'] = reviews.words_in_sentences.apply(sentences_to_vectors)
    reviews['sim_matrix'] = reviews.sentence_vector.apply(similarity_matrix)
    reviews['graph'] = reviews.sim_matrix.apply(compute_graph)
    reviews['summary'] = reviews \
        .apply(lambda d: get_ranked_sentences(d.sentences_in_review, d.graph), axis=1)

    write_file(reviews['summary'].head(1000), 'pos')
    write_file(reviews['summary'].tail(1000), 'neg')


glove_vectors = load_glove_vectors(GLOVE_PATH)
