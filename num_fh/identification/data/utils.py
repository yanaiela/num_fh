import io
import json
import nltk
from nltk.tree import Tree


def read_data(in_f):
    """
    reading the imdb (parsed) corpus
    :param in_f: input file
    :return: yielding a tuple of (text, show-index, scene-index and text-sentence-index)
    """
    with io.open(in_f, 'r', encoding='utf-8') as json_data:
        data = json.load(json_data)
        for show in data:
            show_id = show['id']
            for id_s, scene in enumerate(show['scenes']):
                for id_t, talk in enumerate(scene):
                    if 'meta' in talk: continue
                    text = talk['text']
                    if text.startswith('['):
                        continue
                    yield (text, show_id, id_s, id_t)


def split_sentence(sentence):
    """
    splitting sentence using the nltk model
    :param sentence: sentence to split
    :return: list of split sentences
    """
    sentences = []
    split = nltk.sent_tokenize(sentence)
    for s in split:
        sentences.append(s)

    return sentences


class MyTree(Tree):
    """
    Custom class inhereting from Tree to include indices (by order)
     for the nodes of the tree
    """

    def set_index(self, ind=0):
        if len(self.leaves()) == 1:
            self._i = (ind, ind + 1)
            if isinstance(self[0], MyTree):
                self[0].set_index(ind)
            return ind + 1
        else:
            self._i = (ind, ind + 1)
        for l in self:
            if isinstance(l, unicode):
                return ind
            if len(l.leaves()) == 1:
                ind = l.set_index(ind)
            else:
                ind = l.set_index(ind)
        self._i = (self._i[0], ind)
        return ind

    def get_index(self):
        return self._i


def find_boundaries(s, w):
    """
    find boundaries of a number
    :param s: the nlp'ed sentence
    :param w: the nlp'ed word (number) of the sentence
    :return: start and end indices of the complete number
    """
    ind = w.i
    # handling height
    if ind + 2 < len(s) and s[ind + 1].text == "'" and s[ind + 2].like_num:
        return ind, ind + 3
    if ind - 2 >= 0 and s[ind - 1].text == "'" and s[ind - 2].like_num:
        return ind - 2, ind + 1

    # forward
    if s[ind].ent_iob == 2:
        return ind, ind + 1
    if ind != len(s) - 1:
        i = ind + 1
        while s[i].ent_iob == 1 and (s[i].pos_ == 'NUM' or s[i].like_num or
                                     (i+1 < len(s) and (s[i+1].pos_ == 'NUM' or s[i+1].like_num))):
            i += 1
            if i == len(s):
                break
        if s[i - 1].pos_ == 'NUM' or s[i - 1].like_num or s[i - 1].lemma_ in ['one']:
            end = i
        else:
            end = i - 1
    else:
        end = ind + 1

    # backward
    if s[ind].ent_iob == 3:
        return ind, end
    i = ind - 1
    while s[i].ent_iob != 2 and (s[i].pos_ == 'NUM' or s[i].like_num or s[i-1].pos_ == 'NUM' or s[i-1].like_num):
        i -= 1
        if i == -1:
            break
    i += 1
    if s[i].pos_ != 'NUM' and not s[i].like_num:
        i += 1
    return i, end