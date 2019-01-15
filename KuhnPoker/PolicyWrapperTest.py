import unittest

from KuhnPoker.KuhnPokerGame import KuhnInfoset
from KuhnPoker.PolicyWrapper import infoset_to_state


class InfosetToStateTest(unittest.TestCase):
    def test_game_start_jack(self):
        infoset = KuhnInfoset(0, ())
        state = infoset_to_state(infoset)
        self.assertEqual([1, 0, 0, 0, 0, 0, 0], state.tolist())

    def test_game_start_queen(self):
        infoset = KuhnInfoset(1, ())
        state = infoset_to_state(infoset)
        self.assertEqual([0, 1, 0, 0, 0, 0, 0], state.tolist())

    def test_game_start_king(self):
        infoset = KuhnInfoset(2, ())
        state = infoset_to_state(infoset)
        self.assertEqual([0, 0, 1, 0, 0, 0, 0], state.tolist())

    def test_game_p0_check_jack(self):
        infoset = KuhnInfoset(0, (0,))
        state = infoset_to_state(infoset)
        self.assertEqual([1, 0, 0, 1, 0, 0, 0], state.tolist())

    def test_game_p0_check_king(self):
        infoset = KuhnInfoset(2, (0,))
        state = infoset_to_state(infoset)
        self.assertEqual([0, 0, 1, 1, 0, 0, 0], state.tolist())

    def test_game_p0_bet_jack(self):
        infoset = KuhnInfoset(0, (1,))
        state = infoset_to_state(infoset)
        self.assertEqual([1, 0, 0, 0, 1, 0, 0], state.tolist())

    def test_game_p0_check_p1_bet_jack(self):
        infoset = KuhnInfoset(0, (0, 1))
        state = infoset_to_state(infoset)
        self.assertEqual([1, 0, 0, 1, 0, 0, 1], state.tolist())

    def test_game_p0_check_p1_bet_queen(self):
        infoset = KuhnInfoset(1, (0, 1))
        state = infoset_to_state(infoset)
        self.assertEqual([0, 1, 0, 1, 0, 0, 1], state.tolist())

