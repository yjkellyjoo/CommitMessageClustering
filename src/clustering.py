from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.TextNormalizer import TextNormalizer
from src.KmeansClusters import KMeansClusters
from src.PickledCorpusReader import PickledCorpusReader

if __name__ == "__main__":
    corpus = PickledCorpusReader('../corpus')
    # print(corpus.categories())
    # print(corpus.fileids())

    docs = corpus.docs(categories=['vulnerable'])
    # docs = corpus.docs(categories=['news'])

    # print(docs)

    ## tagging check
    # f = open('../pickled.txt', "w", encoding='UTF-8')
    # list_docs = list(docs)
    # for doc in list_docs:
    #     f.write(str(doc)+'\n')
    #
    # f.close()

    model = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', TfidfVectorizer()),
        ('clusters', KMeansClusters(k=2))
    ])

    clusters = model.fit_transform(docs)


    # print(model.named_steps['norm'])
    #
    # pickles = list(corpus.fileids())
    # for idx, cluster in enumerate(clusters):
    #     print("Document '{}' assigned to cluster {}. ".format(pickles[idx], cluster))

    # num_of_tokens = 0
    # for message in messages:
    #     message = message[0]
    #     tokens = word_tokenize(message)
    #     num_of_tokens = num_of_tokens + len(tokens)
    # print(num_of_tokens)
    #
    # corpus = tfidf.fit_transform(corpus)