import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.TextNormalizer import TextNormalizer
from src.KmeansClusters import KMeansClusters
from src.PickledCorpusReader import PickledCorpusReader
from src.OneHotVectorizer import OneHotVetorizer

from src.constants import *


if __name__ == "__main__":
    corpus = PickledCorpusReader(CORPUS_DIR)
    docs = corpus.docs(categories=CATEGORIES)

    ## tagging check
    # f = open('../pickled.txt', "w", encoding='UTF-8')
    # list_docs = list(docs)
    # for doc in list_docs:
    #     f.write(str(doc)+'\n')
    #
    # f.close()

    ## modeling - KMeansCluster with OneHotVectorizing
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', OneHotVetorizer()),
        ('clusters', KMeansClusters(k=5))
    ])
    clusters = model.fit_transform(docs)

    pickle.dump(clusters, open(KMEANS_MODEL_FILE, 'wb'))

    # print(model.named_steps['norm'])
    #
    # pickles = list(corpus.fileids())
    # for idx, cluster in enumerate(clusters):
    #     print("Document '{}' assigned to cluster {}. ".format(pickles[idx], cluster))

    # corpus = tfidf.fit_transform(corpus)
