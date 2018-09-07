import nltk
from train_data import training_data_generator
import urllib.request
from shifter import shifted_data_generator
from mappings import get_all_mappings, encode
from urllib.error import HTTPError
nltk.download('gutenberg')
nltk.download('reuters')

corpuses = {
    'reuters': nltk.corpus.reuters,
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




