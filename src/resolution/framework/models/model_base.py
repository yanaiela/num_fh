from typing import Dict

from overrides import overrides

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.activations import Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy


@Model.register("nfh_model_base")
class NfhDetector(Model):
    """
    This ``Model`` make a classification of the FH problem.
    Given a sentence and an anchor (number) it create contextualized
    representation for every token and combined with the anchor it
    assign a score, as well to the 6 implicit classes.
    Both the scores for the implicit classes and the tokens are concatenated
    and the highest score is the models' prediction.


    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder: ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder: ``Seq2SeqEncoder``, required
        The encoder that we will use to convert the sentence to a sequence of vectors.
    mlp_dropout: ``float``, required (default = 0.2)
        The dropout probability of the mlp scorer.
    num_implicit: ``int``, required (default = 6)
        The number of implicit classes, additional to the reference ones.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 mlp_dropout: float = 0.2,
                 num_implicit: int = 6) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        # Creating an embedding matrix, holding the parameters for
        # the implicit classes. Each class size is the same as the
        # encoder output.
        self.implicit_embeddings = torch.nn.Parameter(
            torch.Tensor(num_implicit, self.encoder.get_output_dim()))
        torch.nn.init.xavier_uniform_(self.implicit_embeddings)

        # The scorer, an mlp which takes a contextualized class
        # (token or implicit vector) with anchor and assigns a score.
        self.scorer = FeedForward(self.encoder.get_output_dim() * 3, num_layers=2,
                                  hidden_dims=[150, 1], activations=[Activation.by_name('tanh')(),
                                                                     Activation.by_name('linear')()],
                                  dropout=mlp_dropout)

        self._num_implicit = num_implicit

        check_dimensions_match(self.text_field_embedder.get_output_dim(),
                               self.encoder.get_input_dim(),
                               'text embedding dim', 'encoder input dim')
        check_dimensions_match(self.encoder.get_output_dim() * 3,
                               self.scorer.get_input_dim(),
                               '(encoder output dim) * 3', 'scorer input dim')

        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = BooleanAccuracy()

    @overrides
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                anchor_span: torch.Tensor,
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        sentence: Dict[str, torch.Tensor], required
            The input sentence.
        anchor_span: torch.Tensor, required
            The span of the anchor.
        label: torch.Tensor, optional (default = None)
            A variable representing the index of the correct label.
            The first 6 represents the implicit classes, the rest are
            of the reference, representing the indented index of the original
            text.

        Returns
        -------
        An output dictionary consisting of:
        tag_logits: torch.FloatTensor, required
            A tensor of shape ``(batch_size, max_sentence_length)``
            representing a distribution over the label classes for each instance.
        loss: torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        batch_size = anchor_span.shape[0]
        context_token_size = self.encoder.get_output_dim()

        mask = get_text_field_mask(sentence)
        embeddings = self.text_field_embedder(sentence)
        encoder_out = self.encoder(embeddings, mask)

        sentence_len = embeddings.size()[1]

        # Calculates the average embeddings of the anchor spans,
        # in each batch
        anchors = self.get_anchors(anchor_span, encoder_out, batch_size)

        # duplicate the anchor tensor for each batch to the number of
        # tokens in the sentence
        batch_anchors = anchors.unsqueeze(1).expand(batch_size, sentence_len, context_token_size)

        ref_scores = self.calculate_score(batch_anchors, encoder_out)

        implicit_embeddings = self.implicit_embeddings.unsqueeze(0).\
            expand(batch_size, self._num_implicit, context_token_size)

        batch_anchor_implicit = anchors.reshape(batch_size, -1)\
            .unsqueeze(1).expand(batch_size, self._num_implicit, context_token_size)

        implicit_scores = self.calculate_score(batch_anchor_implicit, implicit_embeddings)

        # concatenating all the scores together
        scores = torch.cat([implicit_scores, ref_scores], dim=-1)
        y_hat = torch.argmax(scores, dim=1)

        output = {"tag_logits": scores, "y_hat": y_hat}
        if label is not None:
            self.accuracy(y_hat.reshape(-1), label)
            output["loss"] = self.loss(scores, label)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

    def get_anchors(self,
                    anchor_span: torch.Tensor,
                    encoder_out,
                    batch_size: int):
        anchors = []
        for ind, dim in enumerate(anchor_span):
            anchor_ind_start = dim[0].item()
            anchor_ind_end = dim[-1].item()
            anchor_vec = encoder_out[ind, anchor_ind_start, :]
            for i in range(anchor_ind_start + 1, anchor_ind_end):
                anchor_vec = anchor_vec + encoder_out[ind, i, :]
            anchor_vec = torch.div(anchor_vec, anchor_ind_end - anchor_ind_start + 1)
            anchors.append(anchor_vec)
        anchors = torch.cat(anchors).reshape(batch_size, -1)
        return anchors

    def calculate_score(self, batch_anchor, batch_candidates):
        contextualized_classes = torch.cat([batch_anchor, batch_candidates,
                                                  batch_anchor * batch_candidates], dim=2)
        scores = self.scorer(contextualized_classes)
        scores = scores.reshape(batch_anchor.size()[0], -1)
        return scores
