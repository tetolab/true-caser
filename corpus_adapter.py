from abc import ABC, abstractmethod


class CorpusAdapter(ABC):
    @staticmethod
    @abstractmethod
    def fileids():
        pass

    @staticmethod
    @abstractmethod
    def raw(id=None):
        pass

    @staticmethod
    @abstractmethod
    def sents(id=None):
        pass

    @staticmethod
    @abstractmethod
    def get_train():
        pass

    @staticmethod
    @abstractmethod
    def get_test():
        pass

    @staticmethod
    @abstractmethod
    def get_validation():
        pass