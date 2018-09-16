import numpy as np
from keras.callbacks import Callback


def true_positives(predictions, validation):
    return sum(predictions & validation)


def true_negatives(predictions, validation):
    return sum(~predictions & ~validation)


def false_positives(predictions, validation):
    return sum(predictions & ~validation)


def false_negatives(predictions, validation):
    return sum(~predictions & validation)


def precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)


def recall(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)


def f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))


class F1Score(Callback):
    def on_train_begin(self, logs=None):
        self.precision_hist = list()
        self.recall_hist = list()
        self.f1_hist = list()


    def on_epoch_end(self, epoch, logs=None):
        predictions = np.ndarray.flatten(self.model.predict_classes(self.validation_data[0]))
        predictions = np.ndarray.astype(predictions, np.int64)
        validation = np.ndarray.flatten(self.validation_data[1])
        validation = np.ndarray.astype(validation, np.int64)

        fp = false_positives(predictions, validation)
        tp = true_positives(predictions, validation)
        fn = false_negatives(predictions, validation)

        p = precision(tp, fp)
        r = recall(tp, fn)
        f1 = f1_score(p, r)

        self.precision_hist.append(p)
        self.recall_hist.append(r)
        self.f1_hist.append(f1)

        print(f'- precision: {p} - recall: {r} - f1 score: {f1}')
