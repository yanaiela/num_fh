"""
Usage:
  evaluate.py [--model=MODEL] [--data_dir=DATA_DIR] [--cuda=CUDA]

Options:
  -h --help                     show this help message and exit
  --model=MODEL                 directory of the model
  --data_dir=DATA_DIR           directory of the data
  --cuda=CUDA                   cuda devise to use [default: 0]

"""

import json
from docopt import docopt

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from tqdm import tqdm

from num_fh.resolution.framework.models.model_base import NfhDetector
from num_fh.resolution.framework.dataset_readers.nfh_reader import NFHReader
from num_fh.resolution.framework.dataset_readers.nfh_oracle_reader import NFHReader as OracleNFHReader
from num_fh.resolution.framework.predictors.model_base_predictor import NfhDetectorPredictor


implicit_classes = ['YEAR', 'AGE', 'CURRENCY', 'PEOPLE', 'TIME', 'OTHER']


def get_data(f_name):
    with open(f_name, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    return lines


def get_batched_predictions(predictor, data, limit=10000, bs=20):
    predictions = []
    for i in tqdm(range(1, int(limit / bs))):
        batched_lines = [json.loads(x) for x in data[len(predictions): i * bs]]
        y_hats = predictor.predict_batch_json(batched_lines)
        predictions += y_hats

    batched_lines = [json.loads(x) for x in data[len(predictions): limit]]
    y_hats = predictor.predict_batch_json(batched_lines)
    predictions += y_hats
    return predictions


def get_closest_head(heads, anchors_indices):
    closest_ref = heads[0]
    closest_dist = abs(closest_ref - anchors_indices[0])
    for i in range(len(heads)):
        new_dist = abs(heads[i] - anchors_indices[0])
        if new_dist < closest_dist:
            closest_dist = new_dist
            closest_ref = heads[i]

    return closest_ref


def calc_score(lines, predictions, inclusive=False, sep=None):
    total_correct = 0
    count = 0
    for dev_ex, dev_pred in zip(lines, predictions):
        ex = json.loads(dev_ex)
        y_hat = dev_pred['y_hat']
        if ex['head'][0].__class__ is str:
            if sep == 'ref':
                continue
            if implicit_classes.index(ex['head'][0]) == y_hat:
                total_correct += 1
            count += 1
        else:
            if sep == 'imp':
                continue
            count += 1
            tokens = ex['tokens']
            heads = [tokens[x].lower() for x in ex['head']]

            closest_head = get_closest_head(ex['head'], ex['anchors_indices'])
            if closest_head == y_hat - 6:
                total_correct += 1
            elif inclusive and y_hat >= 6 and tokens[y_hat - 6].lower() in heads:
                total_correct += 1

    return float(total_correct) / count


if __name__ == '__main__':
    arguments = docopt(__doc__)
    model_path = arguments['--model']
    cuda_device = int(arguments['--cuda'])
    archive = load_archive(model_path, cuda_device=cuda_device)
    predictor = Predictor.from_archive(archive, 'nfh_classification')

    data_dir = arguments['--data_dir']

    lines_dev = get_data(data_dir + '/json_dev.jsonl')
    lines_test = get_data(data_dir + '/json_test.jsonl')

    preds_dev = get_batched_predictions(predictor, lines_dev, limit=1000, bs=20)
    preds_test = get_batched_predictions(predictor, lines_test, limit=1000, bs=20)

    for data_split, predictions, name in zip([lines_dev, lines_test], [preds_dev, preds_test], ['dev', 'test']):

        for strict, strict_name in zip([False, True], ['strict', 'inclusive']):
            for split, split_name in zip([None, 'ref', 'imp'], ['all', 'reference', 'implicit']):
                score = calc_score(data_split, predictions, strict, split)
                print(name, strict_name, split_name, score)
