"""
finding fused-heads based on the heuristic discussed in the article
"""
from stanfordcorenlp import StanfordCoreNLP
import spacy
stanford_parser_dir = 'PATH/TO/stanford-corenlp-full-2018-02-27'

stanford_nlp = StanfordCoreNLP(stanford_parser_dir, memory='9g', timeout=15000)
nlp = spacy.load('en')


def substring_sieve(string_list):
    """
    Removing substrings which appears as a larger instance
    :param string_list: list of strings
    :return: filtered list
    """
    string_list.sort(key=lambda s: len(s[0]), reverse=True)
    out = []
    for s in string_list:
        if not any([s[0] in o[0] for o in out]):
            out.append(s)
    return out


def clean(cands):
    """
    performing an additional cleaning on the produced fused-heads
    :param cands: list of possible fh candidates
    :return: "cleaned" list of fh
    """
    fhs = []
    for x in cands:
        fh = []
        ans = stanford_nlp.pos_tag(x[0].encode('utf-8'))
        if len(ans) > 1:
            f, l = None, None
            for ind, (w, p) in enumerate(ans):
                if p not in ['DT', ',', 'PRP', 'IN']:
                    fh.append(w)
                    if f is None:
                        f = x[1][0] + ind
                    l = x[1][0] + ind + 1
            if len(fh) > 0:
                fhs.append((' '.join(fh), (f, l)))
        else:
            if ans[0][1] not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                fhs.append((x[0], x[1]))
    return fhs


def find_fh(tree):
    """
    Given a tree, returning possible fused-heads
    :param tree: a tree, of the MyTree form, based on nltk Tree
    :return:
    """
    # find NPs
    cur = tree
    queue = [cur]
    nps = []
    while len(queue) != 0:
        cur = queue[0]
        queue.pop(0)
        adjs = []
        for child in cur:
            adjs.append(child)
        for adj in adjs:
            if not len(list(adj.subtrees())) == 1:
                queue.append(adj)
            # NX for example: I'll see your *two*... And raise you five.
            if adj.label() in ['NP', 'QP', 'NP-TMP', 'NX', 'SQ']:
                nps.append(adj)

    # find fused-heads in nps
    fhs = []
    for np in nps:
        flag = False
        for child in np:
            # if the NP contain one of those, nothings' missing
            if child.label() in ['NN', 'NNS', 'NNP', 'NNPS', 'NP', 'NX',
                                 'PRP', 'PRP$', 'EX', 'WDT', 'WP', 'WP$', 'WRB', 'POS',
                                 '-LRB-', '-RRB-', '$',
                                 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VP',
                                 'S', 'SBAR',
                                 'FW', 'UH']:
                if len(list(child)) == 1 and nlp(unicode(child.leaves()[0]))[0].like_num:
                    continue
                flag = True

            if len(list(np)) == 1 and child.label() in ['RB', 'CC']:
                flag = True

        # disclude the following list as they cannot serve as fused-head
        if len(np.flatten()) == 1 and np.leaves()[0].lower() in ['no', 'yes', 'the', 'a', 'what']:
            flag = True
        if not flag:
            fhs.append(np)

    # extracting the actual words from the tree
    fhs_text = []
    for fh in fhs:
        fhs_text.append((' '.join(fh.leaves()), fh.get_index()))
    return clean(substring_sieve(fhs_text))

