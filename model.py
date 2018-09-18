from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Activation, InputLayer, Conv1D, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop


def create_model(input_features, classes, units, dropout):
    # First layer inputs must be 3D
    # with shape (samples, timesteps, features)
    model = Sequential()
    model.add(InputLayer(input_shape=(None, input_features)))
    model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout)))
    model.add(Bidirectional(LSTM(units, return_sequences=True, dropout=dropout)))
    model.add(TimeDistributed(Dense(classes)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dense(units))
    # model.add(Activation('relu'))
    # model.add(Dense(classes))
    model.add(Activation('softmax'))
    return model


def compile_model(model):
    model.compile(optimizer=RMSprop(), loss='sparse_categorical_crossentropy', metrics=['accuracy'],
                  sample_weight_mode='temporal')
    return model
