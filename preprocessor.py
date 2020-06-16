import os
import pickle

from nltk.tokenize import wordpunct_tokenize


class preprocessor(object):
    def __init__(self, corpus, target=None):
        self.corpus = corpus
        self.target = target

    def abspath(self, fileid):
        parent = os.path.relpath(
            os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
        )

        basename = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)

        basename = name + '.pickle'

        return os.path.normpath(os.path.join(self.target, parent, basename))

    def tokenize(self, fileid):
        for message in self.corpus.message(fileids=fileid):
            yield wordpunct_tokenize(message)

    def process(self, fileid):
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        if not os.path.exists(parent):
            os.makedirs(parent)

        if not os.path.isdir(parent):
            raise ValueError("Not a directory. Please provide a directory name to write preprocessed data to. ")

        document = list(self.tokenize(fileid))

        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        del document

        return target

