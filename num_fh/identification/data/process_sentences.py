"""
Usage:
  process_sentences.py [--in_file=IN_FILE] [--out_file=OUT_FILE]

Options:
  -h --help                     show this help message and exit
  --in_file=IN_FILE             input json file
  --out_file=OUT_FILE           output file

"""

import io

from docopt import docopt
from tqdm import tqdm

from utils import read_data, split_sentence


def main():
    """
    creates sentences ready for tree parse.
    Getting the json file from the imdb corpus,
     splitting into sentences and saving them into a file
    :return:
    """
    arguments = docopt(__doc__)
    in_file = arguments['--in_file']
    out_file = arguments['--out_file']
    all_sentences = []
    gen = read_data(in_file)

    for entry in tqdm(gen):
        text = entry[0]
        sentences = split_sentence(text)
        for sentence in sentences:
            all_sentences.append((entry[1], entry[2], entry[3], entry[0].text, sentence.text))

    with io.open(out_file, 'w', encoding='utf-8') as f:
        for s in tqdm(all_sentences):
            f.write(s[4] + '\n')


if __name__ == '__main__':
    main()
