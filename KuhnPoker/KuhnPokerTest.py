import unittest
from KuhnPoker import KuhnPokerGame


class IsTerminalTest(unittest.TestCase):
    def test_game_start_is_not_terminal(self):
        bet_sequence = ()
        self.assertFalse(KuhnPokerGame.KuhnNode(bet_sequence).is_terminal)

    def test_first_action_is_not_terminal(self):
        bet_sequence = (0,)
        self.assertFalse(KuhnPokerGame.KuhnNode(bet_sequence).is_terminal)

    def test_p1_folds_is_terminal(self):
        bet_sequence = (1, 0)
        self.assertTrue(KuhnPokerGame.KuhnNode(bet_sequence).is_terminal)

    def test_p1_checks_is_terminal(self):
        bet_sequence = (0, 0)
        self.assertTrue(KuhnPokerGame.KuhnNode(bet_sequence).is_terminal)

    def test_p1_calls_is_not_terminal(self):
        bet_sequence = (0, 1)
        self.assertFalse(KuhnPokerGame.KuhnNode(bet_sequence).is_terminal)

    def test_p0_folds_is_terminal(self):
        bet_sequence = (0, 1, 0)
        self.assertTrue(KuhnPokerGame.KuhnNode(bet_sequence).is_terminal)

    def test_p0_calls_is_terminal(self):
        bet_sequence = (0, 1, 1)
        self.assertTrue(KuhnPokerGame.KuhnNode(bet_sequence).is_terminal)


class PayoffsTest(unittest.TestCase):
    def test_nonterminal_payoff_raises_exception(self):
        with self.assertRaises(RuntimeError):
            KuhnPokerGame.KuhnNode(bet_sequence=()).get_payoffs([0, 0])

    def test_check_check(self):
        bet_sequence = (0, 0)
        payoffs = KuhnPokerGame.KuhnNode(bet_sequence).get_payoffs(player_cards=[0, 1])
        self.assertEqual([-1, 1], payoffs.tolist())

        payoffs = KuhnPokerGame.KuhnNode(bet_sequence).get_payoffs(player_cards=[1, 0])
        self.assertEqual([1, -1], payoffs.tolist())

    def test_bet_fold(self):
        bet_sequence = (1, 0)
        payoffs = KuhnPokerGame.KuhnNode(bet_sequence).get_payoffs(player_cards=[0, 1])
        self.assertEqual([1, -1], payoffs.tolist())

    def test_bet_call(self):
        bet_sequence = (1, 1)
        payoffs = KuhnPokerGame.KuhnNode(bet_sequence).get_payoffs(player_cards=[0, 1])
        self.assertEqual([-2, 2], payoffs.tolist())

        payoffs = KuhnPokerGame.KuhnNode(bet_sequence).get_payoffs(player_cards=[2, 1])
        self.assertEqual([2, -2], payoffs.tolist())

    def test_check_bet_call(self):
        bet_sequence = (0, 1, 1)
        payoffs = KuhnPokerGame.KuhnNode(bet_sequence).get_payoffs(player_cards=[0, 1])
        self.assertEqual([-2, 2], payoffs.tolist())

        bet_sequence = (1, 1)
        payoffs = KuhnPokerGame.KuhnNode(bet_sequence).get_payoffs(player_cards=[2, 1])
        self.assertEqual([2, -2], payoffs.tolist())

    def test_check_bet_fold(self):
        bet_sequence = (0, 1, 0)

        payoffs = KuhnPokerGame.KuhnNode(bet_sequence).get_payoffs(player_cards=[2, 0])
        self.assertEqual([-1, 1], payoffs.tolist())
