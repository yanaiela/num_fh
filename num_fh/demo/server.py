#!/usr/bin/python

import logging
from datetime import datetime

from flask import Flask, request
from flask_cors import CORS, cross_origin
from spacy import displacy

import spacy
from num_fh import NFH


app = Flask(__name__)
CORS(app)

nlp = spacy.load('en_core_web_sm')
nfh = NFH(nlp)
nlp.add_pipe(nfh, first=False)


def get_logger(model_dir):
    time = str(datetime.now()).replace(' ', '-')
    logger = logging.getLogger(time)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(model_dir + '/' + time + '.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger('./logs/')


def add_annotation(doc):
    nfhs = doc._.nfh

    resolved_nfhs = []
    for nfh in nfhs:
        ind = nfh[1]

        start_ind = len(doc[:ind].text_with_ws)
        end_ind = start_ind + len(doc[ind].text_with_ws)

        entity = {'start': start_ind, 'end': end_ind}
        if doc[ind]._.is_implicit:
            label = nfh[2]
            if doc[ind]._.is_deter_nfh:
                label += ' (DETER)'
            entity['label'] = label
            resolved_nfhs.append(entity)
        else:
            label = 'FH'
            entity['label'] = label
            resolved_nfhs.append(entity)

            head = nfh[2]
            head_ind = head.i
            head_start_ind = len(doc[:head_ind].text_with_ws)
            head_end_ind = head_start_ind + len(doc[head_ind].text_with_ws)

            label = 'REFERENCE'
            if doc[ind]._.is_deter_nfh:
                label += ' (DETER)'
            head_entity = {'start': head_start_ind, 'end': head_end_ind, 'label': label}
            resolved_nfhs.append(head_entity)

    # removing duplicate references, based on the starting index
    resolved_nfhs = list({v['start']: v for v in resolved_nfhs}.values())
    return resolved_nfhs


@app.route('/nfh/', methods=['GET'])
@cross_origin()
def serve():
    text = request.args.get('text')
    logger.info('request: ' + text)

    if text.strip() == '':
        return ''

    try:
        doc = nlp(text)
        ans = {'text': doc.text, 'title': None}
        labeled = add_annotation(doc)
        logger.info('ans: ' + str(labeled))
        ans['ents'] = labeled
        html = displacy.render(ans, style="ent", manual=True)
    except Exception as e:
        logger.info('error. ' + str(e))
        html = 'some error occurred while trying to find the NFH'

    return html

