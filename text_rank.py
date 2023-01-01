import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import main
import zipfile


GLOVE_DIR = 'data/glove/'
GLOVE_ZIP = GLOVE_DIR + 'glove.6B.50d.zip'
zip_ref = zipfile.ZipFile(GLOVE_ZIP, 'r')
zip_ref.extractall(GLOVE_DIR)
zip_ref.close()


VECTOR_SIZE = 50
EMPTY_VECTOR = np.zeros(VECTOR_SIZE)

def load_glove_vectors(fn):
    print("Loading Glove Model")
    with open( fn,'r', encoding='utf8') as glove_vector_file:
        model = {}
        for line in glove_vector_file:
            parts = line.split()
            word = parts[0]
            embedding = np.array([float(val) for val in parts[1:]])
            model[word] = embedding
        print("Loaded {} words".format(len(model)))
    return model


def sentence_vector(sentence):
    return sum([glove_vectors.get(word, EMPTY_VECTOR) for word in sentence])/len(sentence)
def sentences_to_vectors(sentences):
    return [sentence_vector(sentence)
        for sentence in sentences]

def similarity_matrix(sentence_vectors):
    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):
            element_i = sentence_vectors[i].reshape(1,VECTOR_SIZE)
            element_j = sentence_vectors[j].reshape(1,VECTOR_SIZE)
            sim_mat[i][j] = cosine_similarity(element_i,
            element_j)[0,0]
    return sim_mat

def compute_graph(sim_matrix):
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    return scores

def get_ranked_sentences(sentences, scores, n=3):
    top_scores = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    top_n_sentences = [sentence for score,sentence in top_scores[:n]]
    return " ".join(top_n_sentences)

reviews = main.reviews_clean_df

glove_vectors = load_glove_vectors('data/glove/glove.6B.50d.txt')
reviews['SimMatrix'] = reviews.SentenceVector.apply(similarity_matrix)
reviews['Graph'] = reviews.SimMatrix.apply(compute_graph)
reviews.head(2)
reviews['Summary'] = reviews.apply(lambda d:get_ranked_sentences(d.SentencesInArticle,d.Graph), axis=1)

