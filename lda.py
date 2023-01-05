from matplotlib import pyplot as plt
import matplotlib.style as style
from gensim import corpora
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel


def calculate_coherence(r, d, m):
    coherence_model = CoherenceModel(model=m, texts=r, dictionary=d, coherence='u_mass')
    return coherence_model.get_coherence()


def get_coherence_values(r, d, c, start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        m = LdaModel(c, num_topics=num_topics, id2word=d, passes=2)
        coherence = calculate_coherence(r, d, m)
        print(f': {coherence}')
        yield coherence


def lda(reviews):
    reviews = reviews.cleaned_text.apply(preprocess_string).tolist()
    dictionary = corpora.Dictionary(reviews)
    corpus = [dictionary.doc2bow(text) for text in reviews]

    # ldamodel = LdaModel(corpus,
    #                     num_topics=NUM_TOPICS,
    #                     id2word=dictionary, passes=15)
    # display(ldamodel.print_topics(num_words=6))

    min_topics, max_topics = 10, 30
    coherence_scores = list(get_coherence_values(reviews, dictionary, corpus, min_topics, max_topics))

    style.use('fivethirtyeight')

    x = [int(i) for i in range(min_topics, max_topics)]

    plt.xticks(x)
    plt.plot(x, coherence_scores)
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence values')
    plt.title('Coherence Scores', fontsize=10)

    plt.show()
