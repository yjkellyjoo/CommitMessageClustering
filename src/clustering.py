import pickle
import json

from src.PickledCorpusReader import PickledCorpusReader
from src.constants import *

if __name__ == "__main__":
    corpus = PickledCorpusReader(CORPUS_DIR)

    pickles = list(corpus.fileids(categories=CATEGORIES))
    kmeans_model = pickle.load(open(KMEANS_MODEL_FILE, 'rb'))

    clusters = {}
    clusters['cluster 1'] = []
    clusters['cluster 2'] = []
    clusters['cluster 3'] = []
    clusters['cluster 4'] = []
    clusters['cluster 5'] = []

    for idx, cluster in enumerate(kmeans_model):
        clusters['cluster ' + str(cluster+1)].append({
            pickles[idx]: list(corpus.sents(pickles[idx]))
        })

    with open('./clusters.json', 'w') as clusters_file:
        json.dump(clusters, clusters_file)
