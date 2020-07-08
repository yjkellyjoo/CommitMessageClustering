from src.lib.Preprocessor import Preprocessor

from src.lib.mysqlModule import *

def preprocessing(table_name, corpus_dir):

    # request query to vulnerable commits
    query = "SELECT message, REPONAME, COMMITID FROM" + table_name + ";"
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
        preprocessor = Preprocessor(message, reponame, commitid, corpus_dir)
        target = preprocessor.transform()
        print(target)

if __name__ == "__main__":
    # connection
    connection = create_connection("121.128.246.13", "git", "gitScraper12#", "33033", "git")

    preprocessing("git.TB_COMMIT", "../corpus/vulnerable")
    preprocessing("git.TB_NOTVULN", "../corpus/not_vulnerable")
