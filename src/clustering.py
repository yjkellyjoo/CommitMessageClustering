from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.TextNormalizer import TextNormalizer
from src.KmeansClusters import KMeansClusters
from src.PickledCorpusReader import PickledCorpusReader

if __name__ == "__main__":
    # TODO:
    corpus = PickledCorpusReader('../corpus')
    docs = corpus.docs(categories=['vulnerable'])

    model = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', TfidfVectorizer()),
        ('clusters', KMeansClusters(k=2))
    ])

    clusters = model.fit_transform(docs)
    pickles = list(corpus.fileids())
    for idx, cluster in enumerate(clusters):
        print("Document '{}' assigned to cluster {}. ".format(pickles[idx], cluster))

    # num_of_tokens = 0
    # for message in messages:
    #     message = message[0]
    #     tokens = word_tokenize(message)
    #     num_of_tokens = num_of_tokens + len(tokens)
    # print(num_of_tokens)
    #
    # corpus = tfidf.fit_transform(corpus)