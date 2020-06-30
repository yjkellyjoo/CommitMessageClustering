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
