import unittest
from itertools import product, repeat
from collections import defaultdict, Counter

import torch

from src.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder(['c', 'a', 't'])
        # real vocab = ['^', 'c', 'a', 't']
        # 1st test without empty token
        probs = torch.tensor([[0, 0.3, 0.5, 0.2], [0, 0.3, 0.3, 0.4]])
        # 1) c: 0.3 a: 0.5 t: 0.2
        # 2) cc: 0.09 ca: 0.09 ct: 0.12 ac: 0.15 aa: 0.15 at: 0.2 ta: 0.06 tc: 0.06, tt:0.08
        # top-3: at, ac, aa. After merge: at, ac, a
        hypos = text_encoder.ctc_beam_search(probs, beam_size=3)
        for hypo in hypos:
            text, prob = hypo
            if text == 'ac':
                self.assertEqual(prob, 0.15)
            elif text == 'a':
                self.assertEqual(prob, 0.15)
            elif text == 'at':
                self.assertEqual(prob, 0.2)
            else:
                raise Exception(f'Wrong hypo with text: {text}, prob: {prob}')

        # 2nd test with merges (repeating letters and accounting for empty token)
        vocab = ['a', 'b']
        text_encoder = CTCCharTextEncoder(vocab)
        probs = torch.tensor([[0, 0.4, 0.6], [0.2, 0.4, 0.4], [0.2, 0.4, 0.4]])

        gts = self.brute_force(text_encoder, probs, ['^'] + vocab)
        hypos = text_encoder.ctc_beam_search(probs, beam_size=27)
        for i in range(len(hypos)):
            (hypo_text, hypo_prob), (gt_text, gt_prob) = hypos[i], gts[i]
            self.assertAlmostEqual(hypo_prob, gt_prob, delta=1e-7)

        # 3rd test: same example, but check that ranking of texts is the same
        # Actually, we're lucky that there'is no pair of texts with the same probability
        # because they could have been ranked differently in gts and hypos
        hypos = text_encoder.ctc_beam_search(probs, beam_size=27)
        self.assertEqual([text for text, _ in hypos], [text for text, _ in gts])

    @staticmethod
    def brute_force(text_encoder, probs, vocab):
        res = []
        for combination in product(*repeat(vocab, len(vocab))):
            combination_prob = 1
            for i, char in enumerate(combination):
                combination_prob *= probs[i][vocab.index(char)]
            res.append((text_encoder.ctc_decode([text_encoder.char2ind[char] for char in combination]),
                        combination_prob))

        res_unique = defaultdict(float)
        for text, prob in res:
            res_unique[text] += prob
        return sorted(res_unique.items(), key=lambda x: x[1], reverse=True)

