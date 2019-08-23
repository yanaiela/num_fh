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

colors = {'REFERENCE': '#428bca',
          'REFERENCE (DETER)': '#e4e7d2',
          'YEAR': '#5cb85c',
          'AGE': '#ffbb33',
          'CURRENCY': '#5bc0de',
          'PEOPLE': '#d9534f',
          'PEOPLE (DETER)': '#d9534f',
          'TIME': '#757575',
          'OTHER': '#e0e0e0',
          'FH': '#c887fb'}
options = {'colors': colors}


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
        nfh_first_token = nfh[0]

        start_ind = len(doc[:nfh_first_token.i].text_with_ws)
        end_ind = start_ind + len(nfh.text_with_ws)

        entity = {'start': start_ind, 'end': end_ind}
        if nfh_first_token._.is_implicit:
            label = nfh_first_token._.nfh_head
            if nfh_first_token._.is_deter_nfh:
                label += ' (DETER)'
            entity['label'] = label
            resolved_nfhs.append(entity)
        else:
            label = 'FH'
            entity['label'] = label
            resolved_nfhs.append(entity)

            head = nfh_first_token._.nfh_head
            head_ind = head.i
            head_start_ind = len(doc[:head_ind].text_with_ws)
            head_end_ind = head_start_ind + len(doc[head_ind].text_with_ws)

            head_label = 'REFERENCE'
            if nfh_first_token._.is_deter_nfh:
                head_label += ' (DETER)'
            head_entity = {'start': head_start_ind, 'end': head_end_ind, 'label': head_label}
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
        logger.info(doc._.nfh)

        labeled = add_annotation(doc)
        # sorting the starting index of the labels, as spacy's renderer expects
        labeled = sorted(labeled, key=lambda k: k['start'])
        logger.info('ans: ' + str(labeled))

        ans = {'text': doc.text, 'title': None,
               'ents': labeled}
        html = displacy.render(ans, style="ent", manual=True, options=options)
    except Exception as e:
        logger.info('error. ' + str(e))
        html = 'some error occurred while trying to find the NFH'

    return html

