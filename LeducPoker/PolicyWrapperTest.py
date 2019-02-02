import unittest

from LeducPoker.LeducPokerGame import LeducInfoset
from LeducPoker.PolicyWrapper import infoset_to_state
from LeducPoker.LeducPokerGame import PlayerActions


class InfosetToStateTest(unittest.TestCase):
    def test_game_start_jack(self):
        infoset = LeducInfoset(0, [(), ()], board_card=None)
        state = infoset_to_state(infoset)
        self.assertEqual([1, 0, 0] + [0] * 19, state.tolist())

    def test_game_start_queen(self):
        infoset = LeducInfoset(1, [(), ()], board_card=None)
        state = infoset_to_state(infoset)
        self.assertEqual([0, 1, 0] + [0] * 19, state.tolist())

    def test_game_start_king(self):
        infoset = LeducInfoset(2, [(), ()], board_card=None)
        state = infoset_to_state(infoset)
        self.assertEqual([0, 0, 1] + [0] * 19, state.tolist())

    def test_game_p0_check_jack(self):
        infoset = LeducInfoset(0, [(PlayerActions.CHECK_CALL,), ()], board_card=None)
        state = infoset_to_state(infoset)
        self.assertEqual(
            [1, 0, 0] +
            [0] * 3 + # Board
            [1, 0] +  # First action
            [0] * 14, state.tolist())

    def test_game_p0_raise_king(self):
        infoset = LeducInfoset(2, [(PlayerActions.BET_RAISE,), ()], board_card=None)
        state = infoset_to_state(infoset)
        self.assertEqual(
            [0, 0, 1] +
            [0] * 3 +  # Board
            [0, 1] +  # First action
            [0] * 14, state.tolist())

    def test_game_p0_crrc_king_queen(self):
        infoset = LeducInfoset(2, [
            (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL), ()],
            board_card=1)
        state = infoset_to_state(infoset)
        self.assertEqual(
            [0, 0, 1] +
            [0, 1, 0] +  # Board
            [1, 0] +  # Action 1
            [0, 1] +  # Action 2
            [0, 1] +  # Action 3
            [1, 0] +  # Action 4
            [0] * 8, state.tolist())

    def test_game_p0_crrc_crrc_king_queen(self):
        infoset = LeducInfoset(2, [
            (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL),
        ], board_card=1)
        state = infoset_to_state(infoset)
        self.assertEqual(
            [0, 0, 1] +
            [0, 1, 0] +  # Board
            [1, 0] +  # Action 1
            [0, 1] +  # Action 2
            [0, 1] +  # Action 3
            [1, 0] +  # Action 4
            [1, 0] +  # Action 1
            [0, 1] +  # Action 2
            [0, 1] +  # Action 3
            [1, 0],  # Action 4
            state.tolist())
