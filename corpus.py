import random

import nltk
import numpy as np
import tensorflow as tf
from more_itertools import flatten
from sklearn.utils import compute_sample_weight

from case_frequency import generate_weights
from mappings import encode, PADDING, gen_input_feature_to_int_map, gen_input_feature_to_class_map
from train_data import training_data_generator
from wiki import WikiAdapter

nltk.download('gutenberg')
nltk.download('reuters')
nltk.download('brown')

corpuses = {
    # 'reuters': ReutersAdapter,
    # 'gutenberg': GutenbergAdapter,
    # 'brown': BrownAdapter,
    'wiki': WikiAdapter
}


def corpus_generator(corpus):
    selected_corpus = corpuses[corpus]

    def func():
        for id in selected_corpus.fileids():
            yield selected_corpus.raw(id)

    return func


def corpus_training_data_generator(corpus, sentence_length, batch_size, shift=True, pad=True):
    for data in training_data_generator(corpus_generator(corpus), sentence_length, batch_size, shift, pad):
        yield data


def split_into_sentences(data, sentence_length):
    sentences = list()
    for text in data:
        current_sentence = ""
        for word in text.split():
            if (len(current_sentence) + len(" ") + len(word)) < sentence_length:
                current_sentence = current_sentence + " " + word
            else:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence != "":
            sentences.append(current_sentence)
    return sentences


def rand_case(c):
    if random.randint(0, 1) == 1 and c != ' ':
        return c.upper()
    else:
        return c.lower()


def randomise_casing(sentences):
    random_cased_sentences = list()
    for sentence in sentences:
        random_cased_sentences.append([rand_case(c) for c in sentence])
    return random_cased_sentences


def lower_casing(sentences):
    lower_cased_sentences = list()
    for sentence in sentences:
        lower_cased_sentences.append([c.lower() for c in sentence])
    return lower_cased_sentences


def tokenize(sentences):
    tokenized_sentences = list()
    for sentence in sentences:
        tokenized_sentences.append(list(sentence))
    return tokenized_sentences


def encode_each_sentence(sentences, mapping):
    encoded_sentences = list()
    for sentence in sentences:
        encoded_sentences.append(encode(sentence, mapping))
    return encoded_sentences


def convert_to_numpy_arrays(sentences):
    return np.asarray(sentences, dtype=np.float32)


def pad(sentences, sentence_length):
    padded_sentences = list()
    for sentence in sentences:
        length_diff = sentence_length - len(sentence)
        if length_diff == 0:
            padded_sentences.append(sentence)
        if length_diff > 0:
            padding = [PADDING] * length_diff
            padded_sentence = sentence
            padded_sentence.extend(padding)
            padded_sentences.append(padded_sentence)
        if length_diff < 0:
            padded_sentences.append(sentence[:sentence_length])
    return padded_sentences


def convert_to_tensors(encoded_sentences):
    return [tf.convert_to_tensor(sentence) for sentence in encoded_sentences]


def generate_sample_weights(tokens):
    weight_map = generate_weights()
    weights = []
    for sentence in tokens:
        weights.append([weight_map.get(token) for token in sentence])
    return weights


def create_all_corpus_train_pipeline(sentence_length, type="train"):
    input_feature_to_int_map = gen_input_feature_to_int_map()
    input_feature_to_class_map = gen_input_feature_to_class_map()
    y = get_texts(type)
    y = split_into_sentences(y, sentence_length)
    y = tokenize(y)
    #w = generate_sample_weights(y)

    #x = randomise_casing(y)
    x = lower_casing(y)
    y = pad(y, sentence_length)
    x = pad(x, sentence_length)
    # w = pad(w, sentence_length)
    w = compute_sample_weight(class_weight='balanced', y=list(flatten(y)))
    x = encode_each_sentence(x, input_feature_to_int_map)
    y = encode_each_sentence(y, input_feature_to_class_map)

    # x = convert_to_tensors(x)
    # y = convert_to_tensors(y)
    x = convert_to_numpy_arrays(x)
    y = convert_to_numpy_arrays(y)
    w = convert_to_numpy_arrays(w)
    w = w.reshape((-1, 100))
    w = np.nan_to_num(w)
    return x, y, w


def get_texts(type):
    texts = list()
    for k, adapter in corpuses.items():
        if type == 'train':
            texts.extend(adapter.get_train())
        elif type == 'validation':
            texts.extend(adapter.get_validation())
        elif type == 'test':
            texts.extend(adapter.get_test())
        else:
            raise ValueError("invalid type")

    return texts
