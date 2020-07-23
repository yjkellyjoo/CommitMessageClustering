import pickle
import time

from sklearn.pipeline import Pipeline

from src.resource.TextNormalizer import TextNormalizer
from src.resource.KmeansClusters import KMeansClusters
from src.resource.PickledCorpusReader import PickledCorpusReader
from src.resource.OneHotVectorizer import OneHotVetorizer
from src.resource.MyTfIdfVectorizer import MyTfidfVectorizer

from src.resource.constants import *

if __name__ == "__main__":
    corpus = PickledCorpusReader(CORPUS_DIR)
    docs = corpus.docs(categories=CATEGORIES)

    ## modeling - KMeansCluster with TfIdfVectorizer
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', MyTfidfVectorizer()),
        ('clusters', KMeansClusters(k=NUMBER_OF_CLUSTERS))
    ])

    run_time = time.time()
    clusters = model.fit_transform(docs)
    run_time = time.time() - run_time
    print("runtime: {}", run_time)

    ## save model into pickle file
    pickle.dump(clusters, open('./output/KMeansCluster_'+str(NUMBER_OF_CLUSTERS)+'.model', 'wb'))

    ## choosing k - checking the sum of squared errors
    ## https://learning.oreilly.com/library/view/k-means-and-hierarchical/9781491965306/ch01.html


    # print(model.named_steps['norm'])
    #
    # pickles = list(corpus.fileids())
    # for idx, cluster in enumerate(clusters):
    #     print("Document '{}' assigned to cluster {}. ".format(pickles[idx], cluster))

    # corpus = tfidf.fit_transform(corpus)
