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


@DatasetReader.register("nfh_reader")
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
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._span_d = self._setting_output_span_indices(1,
                            ['YEAR', 'AGE', 'CURRENCY', 'PEOPLE', 'TIME', 'OTHER'])

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

    def text_to_instance_old(self, scene: DataFrameGroupBy) -> Instance:
        # getting the scene from each group and reindexing inner index in every group

        fields: Dict[str, Field] = {}

        # fields['words_ind'] = scene['index'].tolist()
        # fields['lines_ind'] = scene['line_ind'].tolist()
        # fields['speakers'] = scene['speaker'].tolist()

        target = scene[scene['head'] != '-']
        if len(target) == 0:
            assert False, scene['id'].tolist()[0]

        target_words = scene[scene['head'] != '-']
        assert len(target_words) > 0

        target_span = target_words.index.tolist()

        # sentence = scene['word'].tolist()
        # tar = target_words['word'].tolist()
        # fields['tar'] = target_words['word'].tolist()
        sentence_words = scene['word'].tolist()
        sentence = TextField([Token(t) for t in sentence_words], self._token_indexers)
        fields['sentence'] = sentence
        fields['anchor_span'] = SpanField(target_span[0], target_span[-1], sentence)

        target_class = scene[scene['head'] != '-']['head'].tolist()[0]
        if target_class in self._span_d:  # `Implicit` classes
            head = self._span_d[target_class]
        else:  # `Reference` classes
            ref = scene[scene['referred'] != '-']
            assert len(ref) > 0, scene['id'].tolist()[0]
            l = ref.index.tolist()

            # picking the closest head of the anchor
            head_ref = scene.loc[[l[0]]]['word'].tolist()
            closest_ref = l[0]
            closest_dist = abs(closest_ref - target_words.index[0])
            for i in range(len(l)):
                new_dist = abs(l[i] - target_words.index[0])
                if new_dist < closest_dist:
                    closest_dist = new_dist
                    closest_ref = l[i]

            ref_ind = str(closest_ref) + ':' + str(closest_ref + 1)
            head = self._span_d[ref_ind]

            # fields['head'] = head
            # fields['ref_ind'] = ref_ind
            # fields['head_ref'] = head_ref

        fields['label'] = LabelField(head, skip_indexing=True)
        # if sentence_tags:
        #     fields['sentence_tags'] = SequenceLabelField(sentence_tags, sentence_field)
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
