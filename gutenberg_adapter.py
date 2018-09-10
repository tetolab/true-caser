import nltk

from corpus_adapter import CorpusAdapter

_test_files = [
    'austen-sense.txt',
    'bible-kjv.txt',
    'carroll-alice.txt',
    'chesterton-thursday.txt',
    'shakespeare-caesar.txt', ]


class GutenbergAdapter(CorpusAdapter):

    @staticmethod
    def sents(id=None):
        pass

    @staticmethod
    def fileids():
        return nltk.corpus.gutenberg.fileids()

    @staticmethod
    def raw(id):
        return nltk.corpus.gutenberg.raw(id).split('\n', 1)[1]

    @staticmethod
    def get_train():
        ids = [id for id in GutenbergAdapter.fileids() if id not in _test_files]
        texts = [GutenbergAdapter.raw(id) for id in ids]
        return texts

    @staticmethod
    def get_test():
        ids = [id for id in GutenbergAdapter.fileids() if id in _test_files]
        texts = [GutenbergAdapter.raw(id) for id in ids]
        return texts
