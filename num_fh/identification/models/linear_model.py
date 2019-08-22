""" ***
Usage:
  linear_model.py [--data_dir=DATA_DIR] [--model_out=MODEL_OUT] [--final]

Options:
  -h --help                     show this help message and exit
  --data_dir=DATA_DIR           directory containing pos and neg file
  --model_out=MODEL_OUT         file where to save the trained model
  --final                       final prediction. train on all data

"""
import pickle

import numpy as np
from docopt import docopt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


def get_data(path):
    """
    reading the features dictionary and train and test binaries
    :param path: path to the data directory
    :return: train, dev and test data - both features and labels
    """
    x_train = np.load(path + '/x_train.npy', allow_pickle=True)
    y_train = np.load(path + '/y_train.npy', allow_pickle=True)
    x_dev = np.load(path + '/x_dev.npy', allow_pickle=True)
    y_dev = np.load(path + '/y_dev.npy', allow_pickle=True)
    x_test = np.load(path + '/x_test.npy', allow_pickle=True)
    y_test = np.load(path + '/y_test.npy', allow_pickle=True)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def results(clf, x, y, split):
    """
    print results on data
    :param clf: trained model
    :param x: data
    :param y: correct labels
    :param split - split type (train/dev/test)
    :return:
    """
    y_hat = clf.predict(x)
    print(split, '\t& ', \
        '%.1f' % (100.0 * precision_score(y, y_hat)), '\t& ', \
        '%.1f' % (100.0 * recall_score(y, y_hat)), '\t& ', \
        '%.1f' % (100.0 * f1_score(y, y_hat)), '\\\\')


def inspect(model, x_test, y_test, text_test):
    y_hat = model.predict(x_test)

    for ind in range(x_test.shape[0]):
        if y_hat[ind] != y_test[ind]:
            print(text_test[ind])


if __name__ == '__main__':
    arguments = docopt(__doc__)

    data_d = arguments['--data_dir']
    x_train, y_train, x_dev, y_dev, x_test, y_test = get_data(data_d)

    final = arguments['--final']
    if final:
        x = np.concatenate((x_train, x_dev, x_test))
        y = np.concatenate((y_train, y_dev, y_test))
    else:
        x = x_train
        y = y_train

    clf = Pipeline([
        ('vectorizer', DictVectorizer()),
        ('classifier', LinearSVC())
    ])

    clf.fit(x, y)

    print('& precision 	& recall 	& f1 	\\\\ \hline')
    results(clf, x_train, y_train, 'train')
    results(clf, x_dev, y_dev, 'dev')
    results(clf, x_test, y_test, 'test')

    with open(arguments['--model_out'], 'wb') as handle:
        pickle.dump(clf, handle)
