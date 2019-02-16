import unittest

from LeducPoker.LeducPokerGame import LeducInfoset
from LeducPoker.PolicyWrapper import infoset_to_state_22, infoset_to_state_30
from LeducPoker.LeducPokerGame import PlayerActions
import copy


class InfosetToState30Test(unittest.TestCase):
    def setUp(self):
        self.infosets_seen = set()

    def check_and_insert_infoset(self, infoset):
        infoset_state = tuple(infoset_to_state_30(infoset).tolist())
        self.assertFalse(infoset_state in self.infosets_seen)
        self.infosets_seen.add(infoset_state)

    def check_infoset_unique_recursive(self, infoset: LeducInfoset):
        self.check_and_insert_infoset(infoset)

        if infoset.is_terminal:
            return

        if infoset.player_to_act == -1:
            for card in range(0, 3):
                new_infoset = copy.deepcopy(infoset)
                new_infoset.board_card = card
                self.check_infoset_unique_recursive(new_infoset)
        else:
            check_infoset = copy.deepcopy(infoset)
            check_infoset.add_action(PlayerActions.CHECK_CALL)
            self.check_infoset_unique_recursive(check_infoset)

            if infoset.can_raise:
                raise_infoset = copy.deepcopy(infoset)
                raise_infoset.add_action(PlayerActions.BET_RAISE)
                self.check_infoset_unique_recursive(raise_infoset)

    def test_infosets_unique(self):
        for card in range(0, 3):
            infoset = LeducInfoset(card, [(), ()], board_card=None)
            self.check_and_insert_infoset(infoset)

        self.infosets_seen.clear()

        for card in range(3, 6):
            infoset = LeducInfoset(card, [(), ()], board_card=None)
            self.check_and_insert_infoset(infoset)


class InfosetToState22Test(unittest.TestCase):
    def test_game_start_jack(self):
        infoset = LeducInfoset(0, [(), ()], board_card=None)
        state = infoset_to_state_22(infoset)
        self.assertEqual([1, 0, 0] + [0] * 19, state.tolist())

    def test_game_start_queen(self):
        infoset = LeducInfoset(1, [(), ()], board_card=None)
        state = infoset_to_state_22(infoset)
        self.assertEqual([0, 1, 0] + [0] * 19, state.tolist())

    def test_game_start_king(self):
        infoset = LeducInfoset(2, [(), ()], board_card=None)
        state = infoset_to_state_22(infoset)
        self.assertEqual([0, 0, 1] + [0] * 19, state.tolist())

    def test_game_p0_check_jack(self):
        infoset = LeducInfoset(0, [(PlayerActions.CHECK_CALL,), ()], board_card=None)
        state = infoset_to_state_22(infoset)
        self.assertEqual(
            [1, 0, 0] +
            [0] * 3 + # Board
            [1, 0] +  # First action
            [0] * 14, state.tolist())

    def test_game_p0_raise_king(self):
        infoset = LeducInfoset(2, [(PlayerActions.BET_RAISE,), ()], board_card=None)
        state = infoset_to_state_22(infoset)
        self.assertEqual(
            [0, 0, 1] +
            [0] * 3 +  # Board
            [0, 1] +  # First action
            [0] * 14, state.tolist())

    def test_game_p0_crrc_king_queen(self):
        infoset = LeducInfoset(2, [
            (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL), ()],
            board_card=1)
        state = infoset_to_state_22(infoset)
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
        state = infoset_to_state_22(infoset)
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
