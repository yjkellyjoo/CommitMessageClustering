import nltk
from nltk.cluster import KMeansClusterer

from sklearn.base import BaseEstimator, TransformerMixin


class KMeansClusters(BaseEstimator, TransformerMixin):

    def __init__ (self, k=5):
        """
        :param k: indicating the number of clusters
        """
        self.k = k      # 원하는 군집 수
        self.distance = nltk.cluster.util.cosine_distance       # 선호하는 거리 측정기준
        self.model = KMeansClusterer(self.k, self.distance, avoid_empty_clusters=True)

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        return self.model.cluster(documents, assign_clusters=True)
