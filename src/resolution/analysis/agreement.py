"""
Usage:
  agreement.py [--f=F]

Options:
  -h --help                     show this help message and exit
  --f=F                         file

Code to calculate kappas score.
Based on Vered Shwartz code (thanks!) https://github.com/vered1986

"""

import numpy as np
import pandas
from docopt import docopt
from statsmodels.stats.inter_rater import fleiss_kappa


def compute_majority_gold(results):
    """
    Get the TRUE items that the annotators agreed they are true
    :param results: key to worker answers dictionary
    :return: key to majority label dictionary
    """
    majority_gold = {key: np.argmax(np.bincount([1 if annotations[0] else 0 for worker, annotations
                                                 in results[key].iteritems()]))
                     for key in results.keys()}

    return majority_gold


def load_results(result_file, workers=None, worker_answers=None):
    """
    Load the batch results from the CSV
    :param result_file: the batch results CSV file from MTurk
    :return: the workers and the answers
    """
    if worker_answers is None and workers is None:
        worker_answers = {}
        workers = set()
    table = pandas.read_csv(result_file)

    for index, row in table.iterrows():

        worker_id = row['WorkerId']
        if str(worker_id) == 'nan':
            # worker_id = 'self'
            continue

        # Answer fields
        answer = row['Answer.ans']
        comment = row['Answer.comment']

        key = row['Input.task_id']

        if key not in worker_answers.keys():
            worker_answers[key] = {}

        workers.add(worker_id)

        worker_answers[key][worker_id] = (answer, comment)

    return workers, worker_answers


if __name__ == '__main__':
    arguments = docopt(__doc__)

    batch_result_file1 = arguments['--f']
    workers, results = load_results(batch_result_file1)

    N = len(results)
    k = 7
    n = 3
    label_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6}
    mat = np.zeros((N, k))
    for ind, (_, v) in enumerate(results.items()):
        for a in v.values():
            mat[ind][label_index[a[0]]] += 1
    print(fleiss_kappa(mat))

    full_ag = 0
    part_ag = 0

    for row in mat:
        non_zero = len(np.nonzero(row)[0])
        if non_zero == 1:
            full_ag += 1
        elif non_zero == 2:
            part_ag += 1

    print('full agreement', full_ag)
    print('partial agreement', part_ag)
    print('no agreement', len(mat) - full_ag - part_ag)
