import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Activation, InputLayer


def create_model(input_features, classes, units, dropout):
    # First layer inputs must be 3D
    # with shape (samples, timesteps, features)
    model = Sequential()
    model.add(InputLayer(input_shape=(None, input_features)))
    model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
    model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
    # model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
    # model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)))
    model.add(TimeDistributed(Dense(classes)))
    model.add(Activation('softmax'))
    return model

def compile_model(model):
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model