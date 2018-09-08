from abc import ABC, abstractmethod


class CorpusAdapter(ABC):
    @staticmethod
    @abstractmethod
    def fileids():
        pass

    @staticmethod
    @abstractmethod
    def raw(id):
        pass

    @staticmethod
    @abstractmethod
    def get_train():
        pass

    @staticmethod
    @abstractmethod
    def get_test():
        pass