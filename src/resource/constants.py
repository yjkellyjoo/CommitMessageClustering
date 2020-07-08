PKL_PATTERN = r'(?!\.)[\w_\s]+/[a-zA-Z0-9]+_[a-f0-9]+\.pickle'
PKL_PATTERN_NEWS = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

CORPUS_DIR = '../../corpus'
CATEGORIES = ['vulnerable', 'not_vulnerable']
KMEANS_MODEL_FILE = '../output/KMeansCluster_5.model'