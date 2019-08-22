"""
Usage:
  build_features.py [--data_dir=DATA_DIR] [--out_dir=OUT_DIR] [--window_size=WINDOW] [--no_dep] [--no_pos]

Options:
  -h --help                     show this help message and exit
  --data_dir=DATA_DIR           directory containing pos and neg file
  --out_dir=OUT_DIR             directory for feature vectors
  --window_size=WINDOW          size of the window [default: 3]
  --no_dep                      avoid using dependency features - for ablation test
  --no_pos                      avoid using pos tag features - for ablation test

"""

import numpy as np
import pandas as pd
import spacy
from docopt import docopt

from tqdm import tqdm


class FeatureExtractor(object):

    def __init__(self, window_size, use_dep=True, use_pos=True):
        nlp = spacy.load('en_core_web_sm')
        old_tokenizer = nlp.tokenizer
        nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(self.my_split_function(string))

        self._token_nlp = nlp
        self._window_size = window_size
        self._use_dep = use_dep
        self._use_pos = use_pos

    SEP = u'_SEP_'

    # avoid a different tokenization by spacy
    def my_split_function(self, string):
        return string.split(self.SEP)

    def build_features(self, sentence, inds):
        """
        extracting features for a single example
        :param sentence: list of words
        :param inds: a tuple of the anchor first and last (exclusive) indices
        :return: a dictionary with the features
        """
        feats_dic = {}

        nlp_s = self._token_nlp(self.SEP.join(sentence))
        s_ind = inds[0]
        e_ind = inds[1]

        feats_dic['target'] = ' '.join(sentence[s_ind: e_ind])

        cur_head = nlp_s[s_ind].head
        while cur_head.pos_ in ['NUM']:
            # avoiding endless loop
            if cur_head == cur_head.head:
                break
            cur_head = cur_head.head
        if self._use_dep:
            feats_dic['target_head_'] = cur_head.pos_

        # left window
        for i in range(1, self._window_size + 1):
            if s_ind > i - 1:
                feats_dic['w-{0}'.format(i)] = nlp_s[s_ind - i].lower_
                if self._use_pos:
                    feats_dic['pos-{0}'.format(i)] = nlp_s[s_ind - i].pos_
            else:
                feats_dic['w-{0}'.format(i)] = '^^'

        # right window
        for i in range(self._window_size):
            if e_ind < len(sentence) - i:
                feats_dic['w+{0}_'.format(i + 1)] = nlp_s[e_ind + i].lower_
                if self._use_pos:
                    feats_dic['pos+{0}_'.format(i + 1)] = nlp_s[e_ind + i].pos_
            else:
                feats_dic['w+{0}'.format(i + 1)] = '$$'

        return feats_dic

    def transform_to_dataset(self, data):
        """
        converting the data into features and labels
        :param data: a dataframe containing text and anchor begin and end indices
        :return: list of features and list of binary labels
        """
        X, y = [], []

        for row in tqdm(data[:20]):
            X.append((self.build_features(row[0], row[1:3])))
            y.append(row[3])

        return X, y


def parse_file(in_f):
    """
    reading the file, converting to dataframe and reformatting the text
    :param in_f: input file
    :return: a cleaned dataframe
    """
    df = pd.read_csv(in_f, sep='\t', keep_default_na=False, encoding='utf-8')
    df.text = df.text.apply(lambda text: text.split(FeatureExtractor.SEP))
    return df.values


def main():
    arguments = docopt(__doc__)
    data_f = arguments['--data_dir']
    out_d = arguments['--out_dir']
    window_size = int(arguments['--window_size'])

    train = parse_file(data_f + '/train.tsv')
    dev = parse_file(data_f + '/dev.tsv')
    test = parse_file(data_f + '/test.tsv')

    fe = FeatureExtractor(window_size, not arguments['--no_pos'], not arguments['--no_dep'])

    x_train, y_train = fe.transform_to_dataset(train)
    x_dev, y_dev = fe.transform_to_dataset(dev)
    x_test, y_test = fe.transform_to_dataset(test)

    np.save(out_d + '/x_train.npy', x_train)
    np.save(out_d + '/y_train.npy', y_train)
    np.save(out_d + '/x_dev.npy', x_dev)
    np.save(out_d + '/y_dev.npy', y_dev)
    np.save(out_d + '/x_test.npy', x_test)
    np.save(out_d + '/y_test.npy', y_test)


if __name__ == '__main__':
    main()
