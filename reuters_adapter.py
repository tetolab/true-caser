import nltk

from corpus_adapter import CorpusAdapter


class ReutersAdapter(CorpusAdapter):
    @staticmethod
    def fileids():
        return nltk.corpus.reuters.fileids()

    @staticmethod
    def raw(id):
        return nltk.corpus.reuters.raw(id).split('\n', 1)[1]
