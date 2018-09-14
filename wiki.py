from nltk.corpus import PlaintextCorpusReader

from corpus_adapter import CorpusAdapter

corpus_root = 'data/wiki'
wordlists = PlaintextCorpusReader(corpus_root, '.*')


class WikiAdapter(CorpusAdapter):

    @staticmethod
    def sents(id=None):
        return wordlists.sents(id)
        pass

    @staticmethod
    def fileids():
        return wordlists.fileids()

    @staticmethod
    def raw(id=None):
        return wordlists.raw(id).split('\n', 1)[1]

    @staticmethod
    def get_train():
        sents = WikiAdapter.sents("input.txt")
        return [" ".join(sent) for sent in sents]

    @staticmethod
    def get_validation():
        sents = WikiAdapter.sents("val_input.txt")
        return [" ".join(sent) for sent in sents]

    @staticmethod
    def get_test():
        sents = WikiAdapter.sents("test.txt")
        return [" ".join(sent) for sent in sents]
