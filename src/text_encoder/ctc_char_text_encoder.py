from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        res = ''
        if inds[0]:
            res += self.ind2char[inds[0]]
        for i in range(1, len(inds)):
            new_char = self.ind2char[inds[i]]
            last_char = self.ind2char[inds[i - 1]]
            if new_char != last_char and new_char != self.EMPTY_TOK:
                res += new_char
        return res

    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        dp = {('', self.EMPTY_TOK): 1.}
        for distribution in probs:
            dp = self.extend_and_merge(dp, distribution)
            dp = self.cut_beams(dp, beam_size)

        # need to combine probabilities of the same sentences that were achieved by different paths
        hypos = defaultdict(float)
        for (beam, _), prob in dp.items():
            hypos[beam] += prob

        hypos = [Hypothesis(text, prob) for text, prob in hypos.items()]

        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def extend_and_merge(self, dp, distribution):
        new_dp = defaultdict(float)
        for (beam, last_char), prob in dp.items():
            for char_id, char_prob in enumerate(distribution):
                new_char = self.ind2char[char_id]
                new_beam = beam + new_char if new_char != last_char else beam
                new_beam = new_beam.replace(self.EMPTY_TOK, '')
                new_dp[(new_beam, new_char)] += prob * char_prob
        return new_dp

    @staticmethod
    def cut_beams(dp, beam_size):
        dp = sorted(dp.items(), key=lambda x: x[1], reverse=True)[:beam_size]
        return {k: v for k, v in dp}
