from src.Preprocessor import Preprocessor

from src.mysqlModule import *


if __name__ == "__main__":
    # connection
    connection = create_connection("121.128.246.13", "git", "gitScraper12#", "33033", "git")

    # request query to vulnerable commits
    query = "SELECT message, REPONAME, COMMITID FROM git.TB_COMMIT;"
    results = execute_read_query(connection, query)

    for result in results:
        message = result[0]
        reponame = result[1]
        commitid = result[2]

        #   filter message:
        ##  too short messages cut off
        if len(message) < 50:
            continue

        # tokenize the message and save into pickled file
        preprocessor = Preprocessor(message, reponame, commitid, "../corpus/vulnerable")
        target = preprocessor.transform()
        print(target)

    # corpus = PickledCorpusReader('../corpus')
    # docs = corpus.docs( )
    #
    # model = Pipeline([
    #     ('norm', TextNormalizer()),
    #     ('vect', TfidfVectorizer()),
    #     ('clusters', KmeansClusters(k=2))
    # ])
    #
    # clusters = model.fit_transform(docs)
    # pickles = list(corpus.fileids( ))
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