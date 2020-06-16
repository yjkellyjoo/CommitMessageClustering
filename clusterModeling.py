import KmeansClusters
import TextNormalizer
from mysqlModule import *

from sklearn.feature_extraction.text import TfidfVectorizer

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

        # word_tokenize the messages
        while message is not None:
            for word in word_tokenize(message):
                # preprocess the tokens


            message = cursor.fetchone()
            message = message[0]

    except Error as e:
        print(f"The error '{e}' occurred")



    # num_of_tokens = 0
    # for message in messages:
    #     message = message[0]
    #     tokens = word_tokenize(message)
    #     num_of_tokens = num_of_tokens + len(tokens)
    # print(num_of_tokens)
    #
    # tfidf = TfidfVectorizer()
    # corpus = tfidf.fit_transform(corpus)