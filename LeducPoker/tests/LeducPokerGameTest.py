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
