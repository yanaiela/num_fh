import json
import logging
from typing import Dict, List

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SpanField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides
from pandas.core.groupby import DataFrameGroupBy

logger = logging.getLogger(__name__)


@DatasetReader.register("nfh_orcale_reader")
class NFHReader(DatasetReader):
    """
    This DatasetReader is designed to read a csv file of the nfh dataset.

    It returns a dataset of instances with the following fields:
    sentence : ``TextField``
        The tokens in the sentence.
    anchor_span : ``SpanField``
        The span of the anchor (number)
    label : ``LabelField``
        The label of the instance
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    Returns
    -------
    A ``Dataset`` of ``Instances`` for NFH identification and resolution.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 oracle_head: str = 'ref',
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._span_d = self._setting_output_span_indices(1,
                            ['YEAR', 'AGE', 'CURRENCY', 'PEOPLE', 'TIME', 'OTHER'])
        self._oracle_head = oracle_head

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path), 'r') as f:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                curr_example_json = json.loads(line)
                tokens = curr_example_json['tokens']
                anchors_indices = curr_example_json['anchors_indices']
                head = curr_example_json['head']
                if self._oracle_head == 'ref':
                    if type(head[0]) != int:
                        continue
                else:
                    if type(head[0]) == int:
                        continue

                yield self.text_to_instance(tokens, anchors_indices, head)

    @overrides
    def text_to_instance(self, tokens: List, anchors_indices: List, head: List = None) -> Instance:
        # getting the scene from each group and reindexing inner index in every group

        fields: Dict[str, Field] = {}

        sentence = TextField([Token(t) for t in tokens], self._token_indexers)
        fields['sentence'] = sentence
        fields['anchor_span'] = SpanField(anchors_indices[0], anchors_indices[-1], sentence)

        if head is not None:
            head_first = head[0]
            if head_first in self._span_d:  # `Implicit` classes
                head = self._span_d[head_first]
            else:  # `Reference` classes
                # picking the closest head of the anchor
                closest_ref = head_first
                closest_dist = abs(closest_ref - anchors_indices[0])
                for i in range(len(head)):
                    new_dist = abs(head[i] - anchors_indices[0])
                    if new_dist < closest_dist:
                        closest_dist = new_dist
                        closest_ref = head[i]

                ref_ind = str(closest_ref) + ':' + str(closest_ref + 1)
                head = self._span_d[ref_ind]
            fields['label'] = LabelField(head, skip_indexing=True)

        return Instance(fields)

    def _setting_output_span_indices(self, span_len, additional_classes):
        """
        creating a dictionary from the labels (year, age, etc. and spans indices) to integers
        :param span_len: the maximum possible span length
        :param additional_classes: the `Implicit' classes described in the paper (year, age etc.)
        :return: the mapping dictionary
        """
        span_dic = {}
        counter = 0
        for c in additional_classes:
            span_dic[c] = counter
            counter += 1
        # 10000 is a random large number
        for i in range(10000):
            for j in range(1, span_len + 1):
                s = str(i) + ':' + str(i + j)
                span_dic[s] = counter
                counter += 1

        return dict(span_dic)
