# coding: utf8
from __future__ import unicode_literals

from os import path
import pickle
from typing import List, Tuple, Dict

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from num_fh.api.downloader import download_models
from num_fh.api.downloader import NFH_DIR, IDENTIFICATION_NFH, RESOLUTION_NFH
from num_fh.identification.features.build_features import FeatureExtractor
from num_fh.identification.data.utils import find_boundaries
from num_fh.resolution.framework.models.model_base import NfhDetector
from num_fh.resolution.framework.dataset_readers.nfh_reader import NFHReader
from num_fh.resolution.framework.predictors.model_base_predictor import NfhDetectorPredictor

from spacy.tokens import Doc, Span, Token


IMPLICIT = ['YEAR', 'AGE', 'CURRENCY', 'PEOPLE', 'TIME', 'OTHER']


class NFH(object):
    """spaCy v2.x pipeline component for adding nfh meta data to `Doc` objects.
    Detects Numeric Fused-Heads in `Doc` objects, with two stages:
    Identification and Resolution.
    For every identified NFH, tries to first resolve it deterministically,
    if it does not resolve, move on to the statistical model.

    USAGE:
        >>> import spacy
        >>> from num_fh import NFH
        >>> nlp = spacy.load('en_core_web_sm')
        >>> nfh = NFH(nlp)
        >>> nlp.add_pipe(nfh, first=False)
        >>> doc = nlp("I told you two, that only one of them is the one who will get 2 or 3 icecreams")
        >>> assert doc[16]._.is_nfh == True
        >>> assert doc[18]._.is_nfh == False
        >>> assert doc[3]._.is_deter_nfh == True
        >>> assert doc[16]._.is_deter_nfh == False
        >>> assert len(doc._.nfh) == 4
    """
    name = 'num_fh'

    def __init__(self, nlp, attrs=('has_nfh', 'is_nfh', 'nfh', 'is_deter_nfh',
                                   'nfh_head', 'is_implicit'),
                 force_extension=True):
        """Initialise the pipeline component.

        nlp (Language): The shared nlp object. Used to initialise the matcher
            with the shared `Vocab`, and create `Doc` match patterns.
        RETURNS (callable): A spaCy pipeline component.
        """
        download_models()

        home = path.expanduser("~")

        with open(path.join(home, NFH_DIR, IDENTIFICATION_NFH), 'rb') as f:
            self.identification = pickle.load(f)
            self.feature_extractor = FeatureExtractor(3)

        archive_model = load_archive(path.join(home, NFH_DIR, RESOLUTION_NFH))
        self.resolution_predictor = Predictor.from_archive(archive_model, 'nfh_classification')

        self._has_nfh, self._is_nfh, self._nfh, self._is_deter_nfh, \
            self._nfh_head, self._is_implicit = attrs

        # Add attributes
        Doc.set_extension(self._has_nfh, getter=self.has_nfh, force=force_extension)
        Span.set_extension(self._has_nfh, getter=self.has_nfh, force=force_extension)

        Doc.set_extension(self._nfh, getter=self.iter_nfh, force=force_extension)
        Span.set_extension(self._nfh, getter=self.iter_nfh, force=force_extension)

        Token.set_extension(self._is_nfh, default=False, force=force_extension)
        Token.set_extension(self._is_deter_nfh, default=False, force=force_extension)
        Token.set_extension(self._nfh_head, default=None, force=force_extension)
        Token.set_extension(self._is_implicit, default=False, force=force_extension)

    def __call__(self, doc):
        """Apply the pipeline component to a `Doc` object.

        doc (Doc): The `Doc` returned by the previous pipeline component.
        RETURNS (Doc): The modified `Doc` object.
        """
        fhs = self.identify_nfhs(doc)
        if not fhs:
            return doc

        tokens = [w.text for w in doc]
        for fh_span in fhs:

            # for every fused head found
            span = doc[fh_span[0]: fh_span[1]]
            for token in span:
                token._.set(self._is_nfh, True)

            # deterministic numeric fused-heads
            deter = self.find_deterministic(doc, fh_span)
            if deter:
                for token in span:
                    token._.set(self._is_deter_nfh, True)
                    print(deter)
                    if deter in IMPLICIT:
                        token._.set(self._is_implicit, True)
                        token._.set(self._nfh_head, deter)
                    else:
                        token._.set(self._nfh_head, doc[deter])

            # otherwise, fallback to a statistical model
            else:
                # accounting for the starting token of elmo
                if fh_span[1] - fh_span[0] > 1:
                    fh_fix = [fh_span[0] + 1, fh_span[1] + 1]
                else:
                    fh_fix = [fh_span[0] + 1]
                data = self.data4prediction(tokens, fh_fix)
                ans = self.resolve_prediction(data)
                if ans in IMPLICIT:
                    token._.set(self._is_implicit, True)
                    for token in doc:
                        token._.set(self._nfh_head, ans)
                else:
                    token._.set(self._is_implicit, False)
                    for token in doc:
                        token._.set(self._nfh_head, doc[ans])

        return doc

    @staticmethod
    def find_deterministic(doc, fh_span):
        ind = fh_span[0]

        # Pattern #1 from the paper
        if ind > 0 and doc[ind - 1].lower_ == 'no' and doc[ind].lower_ == 'one':
            return 'PEOPLE'

        # Pattern #2 from the paper
        elif ind > 0 and doc[ind - 1].lower_ == 'you' and doc[ind].lower_ == 'two':
            return 'PEOPLE'
        else:
            # Pattern #3 from the paper
            for ch in doc[ind].children:
                if ch.dep_ == 'prep' and ch.lower_ == 'of':
                    for c in ch.children:
                        if c.dep_ == 'pobj':
                            return c.i

            # Pattern #4 from the paper
            if doc[ind].text.lower() == 'one' and ind > 0 and doc[ind - 1].text.lower() == 'the':
                h = doc[ind].head
                if h.lemma_ == 'be':
                    left_childs = h.lefts
                    for w in left_childs:
                        if w.pos_ in ['PRON', 'PROPN']:
                            return w.i
        return None

    def identify_nfhs(self, doc: Doc) -> List[Tuple[int, int]]:

        numbers = self.find_numbers(doc)
        if len(numbers) == 0:
            return []
        tokens = [w.text for w in doc]
        fh = []
        for num in numbers:
            # feature extraction
            features = self.feature_extractor.build_features(tokens, num)

            # linear model prediction
            y_hat = self.identification.predict(features)
            if y_hat == 1:
                fh.append(num)
        return fh

    @staticmethod
    def find_numbers(doc: Doc) -> List[Tuple[int, int]]:
        numbers = set()
        for i in range(len(doc)):
            if doc[i].pos_ == 'NUM' or doc[i].like_num:
                ind_s, ind_e = find_boundaries(doc, doc[i])
                numbers.add((ind_s, ind_e))
        return list(numbers)

    def resolve_prediction(self, data: Dict):
        val = self.resolution_predictor.predict_json(data)
        ans = val['y_hat']
        # return ans
        if ans < len(IMPLICIT):
            return IMPLICIT[ans]
        else:
            return ans - len(IMPLICIT) - 1

    @staticmethod
    def data4prediction(tokens, num_span):
        json_example = {'tokens': ['<S>'] + tokens + ['</S>'], 'anchors_indices': num_span}
        return json_example

    def has_nfh(self, tokens):
        return any(token._.get(self._is_nfh) for token in tokens)

    def iter_nfh(self, tokens):
        return [(t.text, i, t._.get(self._nfh_head))
                for i, t in enumerate(tokens)
                if t._.get(self._is_nfh)]
