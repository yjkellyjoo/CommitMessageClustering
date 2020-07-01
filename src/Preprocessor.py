import os
import pickle

from nltk import wordpunct_tokenize, sent_tokenize


class Preprocessor(object):
    def __init__(self, message, reponame, commitid, target=None):
        self.message = message
        self.reponame = reponame
        self.fileid = commitid
        self.target = target

    # def fileids(self, fileids=None, categories=None):
    #     fileids = self.corpus.resolve(fileids, categories)
    #     if fileids:
    #         return fileids
    #     return self.corpus.fileids()

    def abspath(self):
        basename = self.reponame + "_" + self.fileid + '.pickle'

        return os.path.normpath(os.path.join(self.target, basename))

    def tokenize(self):
        yield [
            wordpunct_tokenize(sent)
            for sent in sent_tokenize(self.message)
        ]

    def process(self):
        target = self.abspath()
        parent = os.path.dirname(target)

        if not os.path.exists(parent):
            os.makedirs(parent)

        if not os.path.isdir(parent):
            raise ValueError("Not a directory. Please provide a directory name to write preprocessed data to. ")

        document = list(self.tokenize())

        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        del document

        return target
        # return document

    # def transform(self, fileids=None, categories=None):
    #
    #     if not os.path.exists(self.target):
    #         os.makedirs(self.target)
    #
    #     for fileid in self.fileids(fileids, categories):
    #         yield self.process(fileid)

    def transform(self):

        if not os.path.exists(self.target):
            os.makedirs(self.target)

        return self.process()
