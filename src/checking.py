import pickle
import json

from src.resource.PickledCorpusReader import PickledCorpusReader
from src.resource.constants import *

NUMBER_OF_CLUSTERS = 2

if __name__ == "__main__":
    corpus = PickledCorpusReader(CORPUS_DIR)
    pickles = list(corpus.fileids(categories=CATEGORIES))

    ## tagging check
    # f = open('output/pickles.txt', "w", encoding='UTF-8')
    # for one in list(pickles):
    #     f.write(str(one)+'\n')
    #
    # f.close()

    ## clustering check
    kmeans_model = pickle.load(open('./output/KMeansCluster_'+str(NUMBER_OF_CLUSTERS)+'.model', 'rb'))

    clusters = {}

    for i in range(1, NUMBER_OF_CLUSTERS+1):
        clusters['cluster '+str(i)] = []

    for idx, cluster in enumerate(kmeans_model):
        one_pickle = str(pickles[idx])
        pickle_split_tmp = one_pickle.split("/")
        pickle_split_tep = pickle_split_tmp[1].split("_")
        pickle_split = pickle_split_tmp[0], pickle_split_tep[0], pickle_split_tep[1]
        # print(pickle_split)
        clusters['cluster ' + str(cluster+1)].append({
            'name': pickle_split[0],
            'repo': pickle_split[1],
            'commit': pickle_split[2]
        })

    with open('./output/clusters_'+str(NUMBER_OF_CLUSTERS)+'.json', 'w') as clusters_file:
        json.dump(clusters, clusters_file)

