from keras.utils import to_categorical
from more_itertools import flatten

PADDING = 0
UNKNOWN = 1
lowercase_alphabet = 'abcdefghijklmnopqrstuvwxyz'
uppercase_alphabet = lowercase_alphabet.upper()
other_symbols = '1234567890.,- '
all_allowed_symbols = lowercase_alphabet + other_symbols + uppercase_alphabet
all_lower_allowed_symbols = lowercase_alphabet + other_symbols


def gen_mappings(mode='all'):
    mapping = {'UNKNOWN': UNKNOWN, 'PADDING': PADDING}
    length_of_initial_mapping = len(mapping)
    allowed_symbols = all_allowed_symbols
    if mode == 'lower':
        allowed_symbols = all_lower_allowed_symbols
    symbols = sorted(list(set(allowed_symbols)))
    for i, c in enumerate(symbols):
        mapping[c] = i + length_of_initial_mapping
    reverse_mapping = {v: k for k, v in mapping.items()}
    return mapping, reverse_mapping


def get_all_mappings():
    mapping, reverse_mapping = gen_mappings()
    lower_mapping, lower_reverse_mapping = gen_mappings('lower')
    return mapping, reverse_mapping, lower_mapping, lower_reverse_mapping


def map_symbol(mapping, symbol):
    mappings = []
    for t in symbol:
        representation = mapping.get(t)
        if representation:
            mappings.append(representation)
        else:
            mappings.append(UNKNOWN)

    return mappings


def map_dataset(mapping, X):
    X = list(map(lambda word: map_symbol(mapping, word), X))
    X = list(flatten(X))
    return X


def encode(sentence, mapping):
    encoded_sentence = []
    for symbol in sentence:
        if symbol == 0:
            encoded_sentence.append(PADDING)
        elif symbol in mapping:
            encoded_sentence.append(mapping[symbol])
        else:
            encoded_sentence.append(UNKNOWN)
    return to_categorical(encoded_sentence, len(mapping))
