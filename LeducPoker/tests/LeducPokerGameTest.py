import unittest
from LeducPoker import LeducPokerGame
from LeducPoker.LeducPokerGame import PlayerActions

class IsTerminalTest(unittest.TestCase):
    def test_game_start_is_not_terminal(self):
        bet_sequence = ()
        self.assertFalse(LeducPokerGame.LeducNode([bet_sequence, ()], None).is_terminal)

    def test_p1_folds_is_terminal(self):
        bet_sequence = (PlayerActions.BET_RAISE, PlayerActions.FOLD)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], None).is_terminal)

    def test_p2_folds_is_terminal(self):
        bet_sequence = (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.FOLD)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], None).is_terminal)

    def test_checkcall_showdown_is_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        self.assertTrue(LeducPokerGame.LeducNode(bet_sequences, 1).is_terminal)

    def test_raisecall_showdown_is_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)]
        self.assertTrue(LeducPokerGame.LeducNode(bet_sequences, 1).is_terminal)

    def test_checkraise_showdown_is_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)]
        self.assertTrue(LeducPokerGame.LeducNode(bet_sequences, 1).is_terminal)

    def test_checkraise_is_not_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE)]
        self.assertFalse(LeducPokerGame.LeducNode(bet_sequences, 1).is_terminal)

    def test_check_is_not_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL,)]
        self.assertFalse(LeducPokerGame.LeducNode(bet_sequences, 1).is_terminal)

    def test_round1_is_not_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            ()]
        self.assertFalse(LeducPokerGame.LeducNode(bet_sequences, 1).is_terminal)


class PayoffsTest(unittest.TestCase):
    def test_nonterminal_payoff_raises_exception(self):
        with self.assertRaises(RuntimeError):
            LeducPokerGame.LeducNode(bet_sequences=[(),()]).get_payoffs([0, 0])

    def test_check_it_down(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([-1, 1], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([2, 1])
        self.assertEqual([1, -1], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([2, 2])
        self.assertEqual([0, 0], payoffs.tolist())

    def test_board_card_plays(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([0, 2])
        self.assertEqual([1, -1], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([2, 0])
        self.assertEqual([-1, 1], payoffs.tolist())

    def test_bet_fold(self):
        bet_sequences = [(PlayerActions.BET_RAISE, PlayerActions.FOLD), ()]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences).get_payoffs([1, 2])
        self.assertEqual([1, -1], payoffs.tolist())

        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.FOLD), ()]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences).get_payoffs([2, 1])
        self.assertEqual([-1, 1], payoffs.tolist())

    def test_bet_raise_fold(self):
        bet_sequences = [(PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.FOLD), ()]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences).get_payoffs([1, 2])
        self.assertEqual([-3, 3], payoffs.tolist())

        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE,
                                PlayerActions.FOLD), ()]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences).get_payoffs([2, 1])
        self.assertEqual([3, -3], payoffs.tolist())

    def test_round1_bets_are_2(self):
        bet_sequences = [(PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([-3, 3], payoffs.tolist())

    def test_round2_bets_are_4(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([-5, 5], payoffs.tolist())

    def test_round1_bets_are_2_reraise(self):
        bet_sequences = [(PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([-5, 5], payoffs.tolist())

    def test_round2_bets_are_4_reraise(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([-9, 9], payoffs.tolist())

    def test_round1_bets_are_2_check_reraise(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([-5, 5], payoffs.tolist())

    def test_round2_bets_are_4_check_reraise(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE,
                                PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([-9, 9], payoffs.tolist())
