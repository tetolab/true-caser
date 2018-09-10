import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Activation, InputLayer


def create_model(classes, units, dropout):
    # First layer inputs must be 3D
    # with shape (samples, timesteps, features)
    model = Sequential()
    model.add(InputLayer(input_shape=(None, classes)))
    model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
    model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
    model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
    model.add(TimeDistributed(Dense(classes)))
    model.add(Activation('softmax'))
    return model

def compile_model(model):
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model