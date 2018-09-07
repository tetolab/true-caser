import nltk
import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.models import Model, Sequential
from keras.preprocessing.sequence import TimeseriesGenerator, pad_sequences
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from IPython.display import clear_output
from more_itertools import flatten, intersperse
import random
from batcher import batch_from_generator
from train_data import load_conll2003, create_conll_encoded_shifted_generator
from mappings import get_all_mappings

DROPOUT = 0.1
TIME_SLICE_SIZE = 64
BATCH_SIZE = 128
SAMPLING_RATE = 1
PADDING = 0
UNKNOWN = 1
NUM_OF_UNITS = TIME_SLICE_SIZE
WORDS_PER_BATCH = 1000
EPOCHS=1
MODEL_SAVE_PATH = 'tc_model.h5'

mapping, reverse_mapping, lower_mapping, lower_reverse_mapping = get_all_mappings()

def create_model(shape, classes):
    # First layer inputs must be 3D
    # with shape (samples, timesteps, features)
    model = Sequential()
    model.add(Bidirectional(LSTM(NUM_OF_UNITS, batch_input_shape=shape, return_sequences=True, dropout=DROPOUT, recurrent_dropout=DROPOUT)))
    model.add(Bidirectional(LSTM(NUM_OF_UNITS, return_sequences=True, dropout=DROPOUT, recurrent_dropout=DROPOUT)))
    model.add(TimeDistributed(Dense(classes)))
    model.add(Activation('softmax'))
    return model

model = create_model((BATCH_SIZE, TIME_SLICE_SIZE, len(lower_mapping)), len(mapping))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

tensor_board = TensorBoard(batch_size=BATCH_SIZE)

printed_times = 0
for i in range(EPOCHS):
    for data in batch_from_generator(create_conll_encoded_shifted_generator(PADDING, TIME_SLICE_SIZE), BATCH_SIZE * TIME_SLICE_SIZE):
        # hacks 
        if printed_times > 10:
            clear_output()
            print(f'epoch: {i}')
            printed_times = 0
        if len(data) == BATCH_SIZE * TIME_SLICE_SIZE:
            X_train = []
            Y_train = []
            for X, Y in data:
                X_train.extend(X)
                Y_train.extend(Y)
            
            X_train = np.asarray(X_train)
            X_train = np.reshape(X_train, (-1, TIME_SLICE_SIZE, len(lower_mapping)))
            Y_train = np.asarray(Y_train)
            Y_train = np.reshape(Y_train, (-1, TIME_SLICE_SIZE, len(mapping)))

            history = model.fit(X_train, Y_train, batch_size=512,  verbose=2, validation_split=0.2, callbacks=[tensor_board])
            model.save(MODEL_SAVE_PATH)
            printed_times += 1