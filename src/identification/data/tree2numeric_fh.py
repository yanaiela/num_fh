# -*- coding: utf-8 -*-
"""
Usage:
  tree2numeric_fh.py [--imdb_file=IMDB_FILE] [--trees_path=TREES_PATH] [--out_path=OUT_PATH]

Options:
  -h --help                     show this help message and exit
  --imdb_file=IMDB_FILE         imdb json (processed) file
  --trees_path=TREES_PATH       trees path
  --out_path=OUT_PATH           output file

goes over the parsed trees and extract the numeric fused heads. Saving them for training data
This should be ran after parsing all sentences into constituency trees.
"""

import io
import logging

import spacy
from docopt import docopt
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

from find_fh import find_fh
from utils import read_data, MyTree, split_sentence, find_boundaries

nlp = spacy.load('en')
nlp_split = spacy.load('en')

SEP = '##SEP##'


# avoid the default tokenization by spacy
def my_split_function(string):
    return string.split(SEP)


old_tokenizer = nlp_split.tokenizer
nlp_split.tokenizer = lambda string: old_tokenizer.tokens_from_list(my_split_function(string))


def parse_trees_file(in_f):
    """
    Retrieving extracted trees.
    :param in_f: the produced file from stanford-parser process
    :return: a list of constituency trees
    """
    with io.open(in_f, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    trees = []

    flag = False
    tree = ''
    for line in lines:
        if flag:
            if line.strip() != '':
                tree += line
            else:
                trees.append(tree)
                tree = ''
                flag = False
        if line.startswith('Constituency parse:'):
            flag = True
    return trees


def parse_pos_file(in_f):
    """
    Retrieving the pos tags of the sentences
    :param in_f: the produced file from stanford-parser process
    :return: a list of pos tags
    """
    with io.open(in_f, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pos_tags = []

    flag = False
    pos = []
    for line in lines:
        if flag:
            if line.strip() == '':
                pos_tags.append(pos)
                flag = False
                pos = []
            else:
                temp = line.strip().split('PartOfSpeech=')[1][:-1]
                pos.append(temp)
        if line.startswith('Tokens:'):
            flag = True
    return pos_tags


def get_trees(trees_path, num_files=36):
    """
    going over all the parsed sentences files, which were split into different files
    :param trees_path: directory where those files are
    :param num_files: the number of produced files
    :return: list of all the trees (str)
    """
    all_trees = []
    for i in tqdm(range(num_files)):
        if i < 10:
            str_ind = '0' + str(i)
        else:
            str_ind = str(i)
        trees = parse_trees_file(trees_path + 'sentences_' + str_ind + '.out')
        all_trees += trees
    return all_trees


def get_pos_tags(pos_path, num_files=36):
    """
    going over all the parsed sentences files, which were split into different files
    :param pos_path: directory where those files are
    :param num_files: the number of produced files
    :return: list of all the pos tags (str)
    """
    all_pos_tags = []
    for i in tqdm(range(num_files)):
        if i < 10:
            s = '0' + str(i)
        else:
            s = str(i)
        poses = parse_pos_file(pos_path + 'sentences_' + s + '.out')
        all_pos_tags += poses
    return all_pos_tags


def get_imdb_data(imdb_file):
    """
    reading the imdb file and splitting it to sentences
    :param imdb_file: path to the imdb file
    :return: a list of the relevant info from this file
    """
    imdb_data = []
    gen = read_data(imdb_file)
    for entry in tqdm(gen):
        text = entry[0]
        sentences = split_sentence(text)
        for sentence in sentences:
            imdb_data.append((entry[1], entry[2], entry[3], entry[0], sentence))
    return imdb_data


def num_clean(s, inds):
    """
    performing an extensive cleaning for the numeric fh,
     for a specific suspect.
    :param s: list of the sentence token
    :param inds: a tuple of the first and last (exclusive) indices
    :return: (None, None) if it's not a fh, the `cleaned` indices otherwise
    """
    sen = nlp_split(unicode(SEP.join(s)))

    s_ind, e_ind = inds
    # prev token is $ - not fh
    if s_ind > 0:
        if s[s_ind - 1] in [u'$', u'€', u'#']:
            return None, None

    # head of number is noun - not fh
    last_num = sen[s_ind]
    cur_head = last_num.head
    while cur_head.pos_ in ['NUM']:
        # avoiding endless loop
        if cur_head == cur_head.head:
            break
        last_num = cur_head
        cur_head = cur_head.head
    # no missing head
    if last_num.dep_ in ['nummod']:
        return None, None

    # no number at all
    fh_num = nlp(sen[s_ind: e_ind].text)
    flag = False
    for w in fh_num:
        if w.pos_ == 'NUM':
            flag = True
            break
    if not flag:
        return None, None

    if e_ind - s_ind == 1 and '-' in sen[s_ind].text:
        only_num = True
        dash_nlp = nlp_split(unicode(SEP.join(sen[s_ind].text.split('-'))))
        for w in dash_nlp:
            if not w.like_num and w.text.lower() not in ['and', 'a', 'half']:
                only_num = False
        if not only_num:
            return None, None

    if e_ind - s_ind <= 1:
        return s_ind, s_ind + 1
    new_ind_s, new_ind_e = None, None
    flag = True
    for ind in range(e_ind - s_ind):
        num = nlp_split(s[s_ind + ind])[0]
        if num.like_num or num.pos_ == 'NUM':
            if new_ind_s is None:
                new_ind_s = s_ind + ind
            new_ind_e = s_ind + ind + 1
            flag = False
    if flag:
        new_ind_s, new_ind_e = inds
    # cleaning errors of constituency parser
    if new_ind_e < len(sen) and sen[new_ind_e].pos_ in ['NOUN']:
        return None, None
    return new_ind_s, new_ind_e


def get_numeric_fh(trees, imdb):
    """
    creating a list of numeric fused heads out all possible sentences
    :param trees: list of trees (str)
    :param imdb: list of sentences info (from the imdb corpus)
    :return: all extracted fused-heads
    """
    c = 0
    fh_data = []
    pbar = tqdm(zip(trees, imdb))
    for tree, entry in pbar:
        s = ' '.join(tree.split())
        tree = MyTree.fromstring(s)
        tree.set_index()

        fh = find_fh(tree)
        for f in fh:
            v = nlp(f[0].lower())
            for w in v:
                if w.like_num:
                    fh_data.append((entry[0], entry[1], entry[2], entry[3], entry[4], f))
                    c += 1
        pbar.set_description('fh: %i' % c)
    return fh_data


def get_num_ind(tags):
    """
    within a span of suspect fused-head, returning it's numeric boundaries/
    :param tags: list of index and token of suspect fh
    :return: tuple of the relevant indices
    """
    s, e = None, None
    flag = False
    for ind, t in enumerate(tags):
        if t == 'CD':
            if s is None:
                s = ind
                flag = True
            if flag:
                e = ind + 1
        else:
            flag = False
    return s, e


def persist(out_f, data):
    """
    saving numeric like fh examples (positive and negative) to file.
    :param out_f: file where to save the dataset
    :param data: the data to save (list)
    :return:
    """
    with io.open(out_f, 'w', encoding='utf-8') as f:
        for ex_id, entry in tqdm(enumerate(data)):
            ind_s, ind_e = entry[1]
            for ind, w in enumerate(entry[0]):
                f.write(str(ex_id) + '\t' + w + '\t')
                if ind_s <= ind < ind_e:
                    f.write(u'*\n')
                else:
                    f.write(u'-\n')
            f.write(u'\n')


def remove_dups(data, examples):
    """
    removing duplicate examples from the data
    :param data: positive/negative examples: sentences + indices
    :param examples: raw sentences
    :return: subset of the input w/o duplicates
    """
    assert len(data) == len(examples), 'inputs of different size'

    no_dups_data = []
    no_dups_examples = []
    dic = {}

    for row, ex in zip(data, examples):
        flatten = SEP.join(row[0]) + '##FLAT##' + str(row[1][0]) + '#' + str(row[1][1])
        if flatten in dic:
            continue
        dic[flatten] = 1
        no_dups_data.append(row)
        no_dups_examples.append(ex)
    return no_dups_data, no_dups_examples


def create_pos_data(data, parser):
    """
    creating the positive fh numeric dataset. performing another cleaning.
    :param data: suspected fh examples
    :param parser: parser used for the word tokenization
    :return: all positive examples (after the cleaning), will be used
              for creating the negative dataset
    """
    pos_data = []
    pos_examples = []
    for entry in tqdm(data):
        try:
            a = map(unicode, parser.word_tokenize(entry[4].encode('utf-8')))
            s, e = num_clean(a, entry[-1][1])
            if s is not None and (s != entry[-1][1][0] or e != entry[-1][1][1]):
                s, e = num_clean(a, [s, e])
            if s is not None:
                s_nlp = nlp_split(unicode(SEP.join(a)))
                s, e = find_boundaries(s_nlp, s_nlp[s])

                if s >= e:
                    continue

                if s > 0 and (e - s) == 1 and s_nlp[s - 1].pos_ in ['NOUN', 'PROPN'] and s_nlp[s].head == s_nlp[s - 1]:
                    continue

                # time like examples - removing
                if ':' in s_nlp[s:e].text:
                    continue

                # the one token in uppercase is often classified as NOUN
                if s_nlp[s].text.lower() != 'one' and s_nlp[s].pos_ != 'NUM':
                    continue

                pos_data.append((a, (s, e)))

                new_entry = entry[:-1]
                target = (' '.join(a[s:e]), (s, e))
                new_entry = new_entry + (target,)
                pos_examples.append(new_entry)
        except:
            print entry[4]

    pos_data, pos_examples = remove_dups(pos_data, pos_examples)
    return pos_examples, pos_data


def is_fh_pattern(s_nlp, s_ind, e_ind):
    """
    Finding additional patterns, which the constituency parser did not catch
    :param s_nlp: sentence parsed with spaCy
    :param s_ind: start index (inclusive)
    :param e_ind: end index (exclusive)
    :return:
    """
    # not fh - "I need 5 or *6* ounces of dope", not fh - "Well, apart from one or *two* other things"
    # yes fh - "Be good and you'll be back at the mall in a day or *two*, be bad and you won't."
    if s_ind > 0 and s_nlp[s_ind - 1].lemma_ == 'or':
        if e_ind >= len(s_nlp) or (s_nlp[e_ind].pos_ != 'NOUN' and s_nlp[e_ind].head.pos_ != 'NOUN'):
            return True
    # fh - "*eight* or nine clubs"
    if e_ind < len(s_nlp) and s_nlp[e_ind].lemma_ == 'or':
        return True

    # "Appolo *11*'s your secret weapon?", "everyone here lives to their *90*'s, ..."
    if e_ind < len(s_nlp) and s_nlp[e_ind].text.lower() == '\'s':
        if s_ind == 0 or (s_nlp[s_ind - 1].pos_ != 'NOUN' and s_nlp[s_ind - 1].text != '$'):
            return True

    # "... you two ^[NOUN] ..."
    if s_ind > 0 and (e_ind - s_ind == 1) and s_nlp[s_ind].pos_ == 'NUM'\
            and s_nlp[s_ind - 1].text.lower() == 'you':
        if e_ind == len(s_nlp) or (s_nlp[e_ind].pos_ != 'NOUN' and s_nlp[e_ind].head.pos_ != 'NOUN'):
            return True
        if s_nlp[e_ind].pos_ != 'NOUN' and s_nlp[e_ind].head.i < s_nlp[e_ind].i:
            return True

    if s_ind > 0 and e_ind < len(s_nlp) and s_nlp[s_ind - 1].text.lower() == 'the' \
            and s_nlp[s_ind].pos_ == 'NUM' and s_nlp[e_ind].text.lower() == 'who':
        return True

    if e_ind < len(s_nlp) and s_nlp[e_ind].text.lower() == 'of' and s_nlp[s_ind].pos_ == 'NUM':
        return True

    # explicit time
    if e_ind < len(s_nlp) and s_nlp[e_ind].text.lower() in ['a.m', 'p.m', 'a.m.', 'p.m.']:
        return True

    # height: He's at least 6'*5* / *6*'5
    if (e_ind - s_ind == 3) and s_nlp[s_ind + 1].text == "'" and s_nlp[s_ind].like_num and s_nlp[e_ind - 1].like_num:
        return True

    if s_ind > 0 and s_nlp[s_ind - 1].text.lower() == 'no' and (e_ind - s_ind == 1)\
            and s_nlp[s_ind].text.lower() == 'one':
        return True

    return False


# creating negative examples: sentences which contains numbers, and are not a part of the fused-head data
def create_neg_data(all_data, pos_data, pos_tags, parser, out_f):
    """
    creating the negative dataset.
    considering all numbers as candidates, and using all of those
     which weren't part of the positive set.
    :param all_data: all sentences
    :param pos_data: the positive dataset
    :param pos_tags: the pos-tags of all the data
    :param parser: the parsed used for the word tokenization
    :param out_f: output file to save the negative dataset
    :return: all negative examples
    """
    neg_data = []
    neg_examples = []
    extra_pos_data, extra_pos_examples = [], []
    pbar = tqdm(zip(all_data, pos_tags))
    for entry, pos_tag in pbar:
        s = entry[4]
        if 'CD' in pos_tag:
            if s not in pos_data:
                a = parser.word_tokenize(s.encode('utf-8'))
                s_ind, e_ind = get_num_ind(pos_tag)

                # checking for no number
                flag = False
                for i in range(s_ind, e_ind):
                    token = nlp_split(unicode(a[i].lower()))[0]
                    if token.like_num:
                        flag = True
                if not flag:
                    continue

                s_nlp = nlp_split(unicode(SEP.join(a)))
                s_ind, e_ind = find_boundaries(s_nlp, s_nlp[s_ind])

                # some problem
                if s_ind >= e_ind:
                    continue

                if is_fh_pattern(s_nlp, s_ind, e_ind):
                    extra_pos_data.append((a, (s_ind, e_ind)))

                    new_entry = entry
                    target = (' '.join(a[s_ind:e_ind]), (s_ind, e_ind))
                    new_entry = new_entry + (target,)
                    extra_pos_examples.append(new_entry)

                    continue

                neg_data.append((a, (s_ind, e_ind)))
                neg_examples.append(s)

    neg_data, neg_examples = remove_dups(neg_data, neg_examples)
    persist(out_f, neg_data)
    return neg_examples, extra_pos_examples, extra_pos_data


def save_data(data, out_f):
    """
    saving raw data
    :param data: data to save
    :param out_f: output file to save the data
    :return:
    """
    with io.open(out_f, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(row + '\n')


def save_full_info(data, out_f):
    """
        saving raw data with imdb info
        :param data: data to save
        :param out_f: output file to save the data
        :return:
        """
    with io.open(out_f, 'w', encoding='utf-8') as f:
        f.write(u'id\tscene_ind\tsentence_ind\ttext\tsentence\ttarget\tind_s\tind_e\n')
        for row in data:
            tar = row[-1]
            target = tar[0] + '\t' + unicode(tar[1][0]) + '\t' + unicode(tar[1][1])
            f.write(u'\t'.join(map(unicode, row[:-1])) + '\t' + target + u'\n')


def main():
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('creating dataset')

    arguments = docopt(__doc__)
    imdb_file = arguments['--imdb_file']
    out_path = arguments['--out_path']
    trees_path = arguments['--trees_path']

    imdb_data = get_imdb_data(imdb_file)
    trees = get_trees(trees_path)
    all_pos_tags = get_pos_tags(trees_path)
    assert len(trees) == len(imdb_data), 'parsed trees have a different amount of sentences'
    assert len(all_pos_tags) == len(imdb_data), 'parsed sentences have a different amount of sentences'

    fh_data = get_numeric_fh(trees, imdb_data)

    logger.info('extracted %i numeric fused heads' % len(fh_data))

    stanford_nlp = StanfordCoreNLP(r'/home/nlp/lazary/workspace/parsers/stanford-corenlp-full-2018-02-27', memory='9g',
                                   timeout=15000)

    pos_examples, pos_data = create_pos_data(fh_data, stanford_nlp)

    # creating a dictionary with all texts containing fused-heads
    fh_text = {}
    for text in pos_examples:
        fh_text[text[4]] = 1

    neg_examples, extra_pos_ex, extra_pos_data = create_neg_data(imdb_data, fh_text, all_pos_tags, stanford_nlp,
                                                                 out_path + '/neg')

    all_pos_data = pos_data + extra_pos_data
    all_pos_ex = pos_examples + extra_pos_ex
    all_pos_data, all_pos_ex = remove_dups(all_pos_data, all_pos_ex)
    persist(out_path + '/pos', all_pos_data)

    save_data([x[4] for x in all_pos_ex], out_path + '/raw_pos')
    save_data(neg_examples, out_path + '/raw_neg')

    save_full_info(all_pos_ex, out_path + '/full_pos')


if __name__ == '__main__':
    main()
