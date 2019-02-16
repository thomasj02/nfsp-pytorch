import unittest
from LeducPoker import LeducPokerGame
from LeducPoker.LeducPokerGame import PlayerActions

class IsTerminalTest(unittest.TestCase):
    def test_game_start_is_not_terminal(self):
        bet_sequence = ()
        self.assertFalse(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).is_terminal)

    def test_p1_folds_is_terminal(self):
        bet_sequence = (PlayerActions.BET_RAISE, PlayerActions.FOLD)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).is_terminal)

    def test_p2_folds_is_terminal(self):
        bet_sequence = (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.FOLD)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).is_terminal)

    def test_checkcall_showdown_is_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        self.assertTrue(LeducPokerGame.LeducNode(bet_sequences, board_card=1).is_terminal)

    def test_raisecall_showdown_is_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)]
        self.assertTrue(LeducPokerGame.LeducNode(bet_sequences, board_card=1).is_terminal)

    def test_checkraise_showdown_is_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)]
        self.assertTrue(LeducPokerGame.LeducNode(bet_sequences, board_card=1).is_terminal)

    def test_checkraise_is_not_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE)]
        self.assertFalse(LeducPokerGame.LeducNode(bet_sequences, board_card=1).is_terminal)

    def test_check_is_not_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.CHECK_CALL,)]
        self.assertFalse(LeducPokerGame.LeducNode(bet_sequences, board_card=1).is_terminal)

    def test_round1_is_not_terminal(self):
        bet_sequences = [
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            ()]
        self.assertFalse(LeducPokerGame.LeducNode(bet_sequences, board_card=1).is_terminal)


class GameRoundTest(unittest.TestCase):
    def test_game_start_is_round_0(self):
        bet_sequence = ()
        self.assertEqual(0, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)

    def test_game_c_is_round_0(self):
        bet_sequence = (PlayerActions.CHECK_CALL,)
        self.assertEqual(0, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)

    def test_game_r_is_round_0(self):
        bet_sequence = (PlayerActions.BET_RAISE,)
        self.assertEqual(0, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)

    def test_game_cc_is_round_1(self):
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        self.assertEqual(1, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)

    def test_game_cr_is_round_0(self):
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE)
        self.assertEqual(0, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)

    def test_game_rr_is_round_0(self):
        bet_sequence = (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE)
        self.assertEqual(0, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)

    def test_game_crrc_is_round_1(self):
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE,
                        PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)
        self.assertEqual(1, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)

    def test_game_crr_is_round_0(self):
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE)
        self.assertEqual(0, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)

    def test_game_rrc_is_round_1(self):
        bet_sequence = (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)
        self.assertEqual(1, LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).game_round)


class CanFoldTest(unittest.TestCase):
    def test_cannot_open_fold(self):
        bet_sequence = ()
        self.assertFalse(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_fold)

    def test_cannot_fold_after_c(self):
        bet_sequence = (PlayerActions.CHECK_CALL,)
        self.assertFalse(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_fold)

    def test_can_fold_after_r(self):
        bet_sequence = (PlayerActions.BET_RAISE,)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_fold)

    def test_can_fold_after_cr(self):
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_fold)

    def test_cannot_open_fold_r2(self):
        r1_bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = ()
        self.assertFalse(LeducPokerGame.LeducNode([r1_bet_sequence, bet_sequence], board_card=0).can_fold)

    def test_cannot_fold_after_c_r2(self):
        r1_bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = (PlayerActions.CHECK_CALL,)
        self.assertFalse(LeducPokerGame.LeducNode([r1_bet_sequence, bet_sequence], board_card=0).can_fold)

    def test_can_fold_after_r_r2(self):
        r1_bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = (PlayerActions.BET_RAISE,)
        self.assertTrue(LeducPokerGame.LeducNode([r1_bet_sequence, bet_sequence], board_card=0).can_fold)

    def test_can_fold_after_cr_r2(self):
        r1_bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE)
        self.assertTrue(LeducPokerGame.LeducNode([r1_bet_sequence, bet_sequence], board_card=0).can_fold)


class CanRaiseTest(unittest.TestCase):
    def test_can_open_raise(self):
        bet_sequence = ()
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_raise)

    def test_can_raise_after_c(self):
        bet_sequence = (PlayerActions.CHECK_CALL,)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_raise)

    def test_can_raise_after_r(self):
        bet_sequence = (PlayerActions.BET_RAISE,)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_raise)

    def test_can_raise_after_cr(self):
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE)
        self.assertTrue(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_raise)

    def test_cannot_raise_after_rr(self):
        bet_sequence = (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE)
        self.assertFalse(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_raise)

    def test_cannot_raise_after_crr(self):
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE)
        self.assertFalse(LeducPokerGame.LeducNode([bet_sequence, ()], board_card=0).can_raise)

    def test_r2_can_open_raise(self):
        r1_actions = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = ()
        self.assertTrue(LeducPokerGame.LeducNode([r1_actions, bet_sequence], board_card=0).can_raise)

    def test_r2_can_raise_after_c(self):
        r1_actions = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = (PlayerActions.CHECK_CALL,)
        self.assertTrue(LeducPokerGame.LeducNode([r1_actions, bet_sequence], board_card=0).can_raise)

    def test_r2_can_raise_after_r(self):
        r1_actions = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = (PlayerActions.BET_RAISE,)
        self.assertTrue(LeducPokerGame.LeducNode([r1_actions, bet_sequence], board_card=0).can_raise)

    def test_r2_can_raise_after_cr(self):
        r1_actions = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE)
        self.assertTrue(LeducPokerGame.LeducNode([r1_actions, bet_sequence], board_card=0).can_raise)

    def test_r2_cannot_raise_after_rr(self):
        r1_actions = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE)
        self.assertFalse(LeducPokerGame.LeducNode([r1_actions, bet_sequence], board_card=0).can_raise)

    def test_r2_cannot_raise_after_crr(self):
        r1_actions = (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)
        bet_sequence = (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE)
        self.assertFalse(LeducPokerGame.LeducNode([r1_actions, bet_sequence], board_card=0).can_raise)


class AddActionTest(unittest.TestCase):
    def test_first_check_is_antes(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[(),()], board_card=None).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(1, cost)

    def test_first_bet(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[(),()], board_card=None).add_action(PlayerActions.BET_RAISE)
        self.assertEqual(3, cost)

    def test_raise_fold(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[(PlayerActions.BET_RAISE,),()],
                                        board_card=None).add_action(PlayerActions.FOLD)
        self.assertEqual(1, cost)

    def test_cc_r1(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[(PlayerActions.CHECK_CALL,),()],
                                        board_card=None).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(1, cost)

    def test_call_r1(self):
        cost = LeducPokerGame.LeducNode(
            bet_sequences=[(PlayerActions.BET_RAISE,),()], board_card=None).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(3, cost)

    def test_reraise_r1(self):
        cost = LeducPokerGame.LeducNode(
            bet_sequences=[(PlayerActions.BET_RAISE,),()], board_card=None).add_action(PlayerActions.BET_RAISE)
        self.assertEqual(5, cost)

    def test_call_reraise_r1(self):
        cost = LeducPokerGame.LeducNode(
            bet_sequences=[
                (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE),
                ()], board_card=None).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(2, cost)

    def test_final_call_r1(self):
        cost = LeducPokerGame.LeducNode(
            bet_sequences=[
                (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE),
                ()], board_card=None).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(2, cost)

    def test_first_check_is_free_r2(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            ()], board_card=0).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(0, cost)

    def test_first_bet_r2(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            ()], board_card=0).add_action(PlayerActions.BET_RAISE)
        self.assertEqual(4, cost)

    def test_call_r2(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.BET_RAISE,)], board_card=0).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(4, cost)

    def test_reraise_r2(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.BET_RAISE,)], board_card=0).add_action(PlayerActions.BET_RAISE)
        self.assertEqual(8, cost)

    def test_call_reraise_r2(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE)], board_card=0).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(4, cost)

    def test_final_call_r2(self):
        cost = LeducPokerGame.LeducNode(bet_sequences=[
            (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
            (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE)],
            board_card=0).add_action(PlayerActions.CHECK_CALL)
        self.assertEqual(4, cost)


class PayoffsTest(unittest.TestCase):
    def test_nonterminal_payoff_raises_exception(self):
        with self.assertRaises(RuntimeError):
            LeducPokerGame.LeducNode(bet_sequences=[(),()], board_card=0).get_payoffs([0, 0])

    def test_check_it_down(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([0, 2], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([2, 1])
        self.assertEqual([2, 0], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([2, 2])
        self.assertEqual([1, 1], payoffs.tolist())

    def test_board_card_plays(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([0, 2])
        self.assertEqual([2, 0], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([2, 0])
        self.assertEqual([0, 2], payoffs.tolist())

    def test_suits_are_equivalent(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([3, 2])
        self.assertEqual([2, 0], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=3).get_payoffs([0, 2])
        self.assertEqual([2, 0], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=2).get_payoffs([0, 5])
        self.assertEqual([0, 2], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=5).get_payoffs([0, 2])
        self.assertEqual([0, 2], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=1).get_payoffs([0, 4])
        self.assertEqual([0, 2], payoffs.tolist())

        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=4).get_payoffs([0, 1])
        self.assertEqual([0, 2], payoffs.tolist())

    def test_bet_fold(self):
        bet_sequences = [(PlayerActions.BET_RAISE, PlayerActions.FOLD), ()]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([2, 0], payoffs.tolist())

        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.FOLD), ()]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([2, 1])
        self.assertEqual([0, 2], payoffs.tolist())

    def test_bet_raise_fold(self):
        bet_sequences = [(PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.FOLD), ()]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([0, 6], payoffs.tolist())

        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE,
                                PlayerActions.FOLD), ()]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([2, 1])
        self.assertEqual([6, 0], payoffs.tolist())

    def test_round1_bets_are_2(self):
        bet_sequences = [(PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([0, 6], payoffs.tolist())

    def test_round2_bets_are_4(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([0, 10], payoffs.tolist())

    def test_round1_bets_are_2_reraise(self):
        bet_sequences = [(PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([0, 10], payoffs.tolist())

    def test_round2_bets_are_4_reraise(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([0, 18], payoffs.tolist())

    def test_round1_bets_are_2_check_reraise(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([0, 10], payoffs.tolist())

    def test_round2_bets_are_4_check_reraise(self):
        bet_sequences = [(PlayerActions.CHECK_CALL, PlayerActions.CHECK_CALL),
                         (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.BET_RAISE,
                                PlayerActions.CHECK_CALL)]
        payoffs = LeducPokerGame.LeducNode(bet_sequences=bet_sequences, board_card=0).get_payoffs([1, 2])
        self.assertEqual([0, 18], payoffs.tolist())
