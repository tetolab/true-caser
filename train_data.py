import random

import nltk
import numpy as np
from keras.utils import to_categorical
from more_itertools import intersperse

from batcher import batch
from mappings import encode
from mappings import gen_mappings, get_all_mappings, map_dataset
from padder import pad as pad_sentence
from shifter import generate_shifted_data
from shifter import shifted_data_generator

conll_train_path = './data/conll2003/en/train.txt'


def rand_case(c):
    if random.randint(0, 1) == 1:
        return c.upper()
    else:
        return c.lower()


def encode_x_y(x, x_mapping, y, y_mapping):
    try:
        x = encode([rand_case(c) for c in x], x_mapping)
        y = encode(y, y_mapping)
        return x, y
    except AttributeError as err:
        print(x, y)
        pass


def training_data_generator(text_generator, sentence_length, batch_size, shift=True, pad=True):
    mapping, _, lower_mapping, _ = get_all_mappings()
    X = []
    Y = []
    for text in text_generator():
        for sentence in batch(text, sentence_length):
            try:
                if shift:
                    for shifted_sentence in shifted_data_generator(sentence, ' ', sentence_length):
                        x, y = encode_x_y(shifted_sentence, mapping, shifted_sentence, mapping)
                        X.append(x)
                        Y.append(y)
                        if len(X) >= batch_size:
                            X = np.asarray(X).reshape(-1, sentence_length, len(mapping))
                            Y = np.asarray(Y).reshape(-1, sentence_length, len(mapping))
                            yield X, Y
                            X = []
                            Y = []
                else:
                    if pad:
                        sentence = pad_sentence(sentence, sentence_length, ' ')
                    x, y = encode_x_y(sentence, mapping, sentence, mapping)
                    X.append(x)
                    Y.append(y)
                    if len(X) >= batch_size:
                        X = np.asarray(X).reshape(-1, len(sentence), len(mapping))
                        Y = np.asarray(Y).reshape(-1, len(sentence), len(mapping))
                        yield X, Y
                        X = []
                        Y = []
            except ValueError as err:
                print('Error: ', err)
    # yield X, Y


def get_emma():
    return nltk.corpus.gutenberg.words('austen-emma.txt')


def get_nltk_gutenberg(name):
    mapping, reverse_mapping = gen_mappings()
    lower_mapping, lower_reverse_mapping = gen_mappings('lower')
    X = nltk.corpus.gutenberg.words(name)
    X = list(intersperse(' ', X))
    Y = X
    X = [x.lower() for x in X]
    X = map_dataset(lower_mapping, X)
    Y = map_dataset(mapping, Y)
    Y = to_categorical(Y, len(mapping))
    X = to_categorical(X, len(lower_mapping))
    return list(zip(X, Y)), mapping, reverse_mapping, lower_mapping, lower_reverse_mapping


def load_conll2003():
    sentences = []

    with open(conll_train_path, 'r') as file:
        current_sentence = []
        for line in file:
            parsed_line = parse_line(line)
            if parsed_line == 'START OF DOC':
                pass
            elif parsed_line == 'END SENTENCE':
                if len(current_sentence) > 0:
                    sentences.append(current_sentence)
                current_sentence = []
            else:
                current_sentence.append(parsed_line)

    return sentences


def parse_line(line):
    if '-DOCSTART-' in line:
        return 'START OF DOC'
    elif line == '\n':
        return 'END SENTENCE'
    else:
        return line.split()[0]


def conll_sentence_generator(with_spaces=True, padded=True, padding_size=32):
    with open(conll_train_path, 'r') as file:
        current_sentence = []
        for line in file:
            parsed_line = parse_line(line)
            if parsed_line == 'START OF DOC':
                pass
            elif parsed_line == 'END SENTENCE':
                if len(current_sentence) > 0:
                    if with_spaces:
                        current_sentence = list(intersperse(' ', current_sentence))
                    yield current_sentence
                current_sentence = []
            else:
                current_sentence.append(parsed_line)


def conll_shifted_sentence_generator(padding_symbol=0, max_shift_len=32):
    for sentence in conll_sentence_generator():
        for shifted_sentence in generate_shifted_data(sentence, max_len_of_sentence=max_shift_len,
                                                      padding_symbol=padding_symbol):
            yield shifted_sentence


def create_conll_encoded_shifted_generator(padding_symbol=0, batch_size=32, max_shift_len=32):
    return lambda: conll_encoded_shifted_generator(padding_symbol=padding_symbol, max_shift_len=max_shift_len)


def conll_encoded_shifted_generator(padding_symbol=0, batch_size=32, max_shift_len=32):
    mapping, _, lower_mapping, _ = get_all_mappings()
    for sentence in conll_shifted_sentence_generator(padding_symbol=padding_symbol, max_shift_len=max_shift_len):
        X = encode(sentence, lower_mapping)
        Y = encode(sentence, mapping)
        yield (X, Y)
