from overrides import overrides

from allennlp.data import Instance
from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor


@Predictor.register('nfh_classification')
class NfhDetectorPredictor(Predictor):
    """"Predictor wrapper for the NfhDetector"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
    # def _json_to_instance(self, json_dict: JsonDict) -> JsonDict:
        sentence = json_dict['tokens']
        anchor_span = json_dict['anchors_indices']
        label = json_dict['label'] if 'label' in json_dict else None
        instance = self._dataset_reader.text_to_instance(tokens=sentence, anchors_indices=anchor_span, head=label)

        # span_d = self._setting_output_span_indices(1,
        #                                           ['YEAR', 'AGE', 'CURRENCY', 'PEOPLE', 'TIME', 'OTHER'])
        # label_dict = {v: k for k, v in span_d.items()}

        #return {'instance': instance, 'label_dict': label_dict}
        return instance

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
