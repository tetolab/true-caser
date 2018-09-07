import nltk

from reuters_adapter import ReutersAdapter
from train_data import training_data_generator

nltk.download('gutenberg')
nltk.download('reuters')

corpuses = {
    'reuters': ReutersAdapter,
    'gutenberg': nltk.corpus.gutenberg
}


def corpus_generator(corpus):
    selected_corpus = corpuses[corpus]

    def func():
        for id in selected_corpus.fileids():
            yield selected_corpus.raw(id)

    return func


def corpus_training_data_generator(corpus, sentence_length, batch_size, shift=True):
    for data in training_data_generator(corpus_generator(corpus), sentence_length, batch_size, shift):
        yield data
