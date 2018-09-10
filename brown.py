import nltk
import tensorflow as tf
from corpus_adapter import CorpusAdapter


class BrownAdapter(CorpusAdapter):

    @staticmethod
    def sents(id=None):
        return nltk.corpus.brown.sents(id)
        pass

    @staticmethod
    def fileids():
        return nltk.corpus.brown.fileids()

    @staticmethod
    def raw(id=None):
        return nltk.corpus.brown.raw(id).split('\n', 1)[1]

    @staticmethod
    def get_train():
        ids = BrownAdapter.fileids()
        sents = [BrownAdapter.sents(id) for id in ids]
        return [" ".join(z) for z in [[" ".join(y) for y in x] for x in sents]]

    @staticmethod
    def get_test():
        raise NotImplementedError

    @staticmethod
    def get_train_dataset():
        ids = BrownAdapter.fileids()




