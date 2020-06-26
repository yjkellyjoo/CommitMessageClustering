import KmeansClusters
import TextNormalizer
import Preprocessor
import PickledCorpusReader
from mysqlModule import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import json


if __name__ == "__main__":
    # connection
    connection = create_connection("121.128.246.13", "git", "gitScraper12#", "33033", "git")

    # request query
    query = "SELECT message FROM git.TB_COMMIT;"

    cursor = connection.cursor()
    try:
        cursor.execute(query)
        message = cursor.fetchone()
        message = message[0]

        while message is not None:
            # TODO: tokenize the messages
            Preprocessor.Preprocessor.tokenize(message)

            message = cursor.fetchone()
            message = message[0]

    except Error as e:
        print(f"The error '{e}' occurred")


    corpus = PickledCorpusReader('../corpus')
    docs = corpus.docs( )

    model = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', TfidfVectorizer()),
        ('clusters', KmeansClusters(k=2))
    ])

    clusters = model.fit_transform(docs)
    pickles = list(corpus.fileids( ))
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