import nltk

from corpus_adapter import CorpusAdapter


class ReutersAdapter(CorpusAdapter):

    @staticmethod
    def fileids():
        return nltk.corpus.reuters.fileids()

    @staticmethod
    def raw(id):
        return nltk.corpus.reuters.raw(id).split('\n', 1)[1]

    @staticmethod
    def get_train():
        ids = [id for id in ReutersAdapter.fileids() if 'train' in id]
        texts = [ReutersAdapter.raw(id) for id in ids]
        return texts

    @staticmethod
    def get_test():
        ids = [id for id in ReutersAdapter.fileids() if 'test' in id]
        texts = [ReutersAdapter.raw(id) for id in ids]
        return texts
