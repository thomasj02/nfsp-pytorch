from typing import Tuple, Optional, List
import random
import numpy as np


class PlayerActions:
    BET_RAISE = 2
    CHECK_CALL = 1
    FOLD = 0


class LeducNode(object):
    def __init__(
            self,
            bet_sequences: List[Tuple],
            hidden_board_card: Optional[int] = None,
            board_card: Optional[int] = None):
        assert len(bet_sequences) == 2
        assert (hidden_board_card is not None) or (board_card is not None)
        assert len(bet_sequences[1]) == 0 or (board_card is not None)  # On first round, or board card showing
        assert len(bet_sequences[1]) > 0 or (board_card is None)  # On second round, or board card hidden
        self._bet_sequences = bet_sequences
        self._hidden_board_card = hidden_board_card
        self.board_card = board_card
        self.game_round = 0

    @property
    def bet_sequences(self):
        return self._bet_sequences

    @property
    def is_terminal(self):
        if len(self._bet_sequences[0]) > 0 and self._bet_sequences[0][-1] == PlayerActions.FOLD:
            return True

        if len(self._bet_sequences[1]) <= 1:
            return False

        if self._bet_sequences[1][-1] != PlayerActions.BET_RAISE:
            return True

        return False

    @property
    def player_to_act(self):
        return (len(self._bet_sequences[0]) + len(self._bet_sequences[1])) % 2

    def add_action(self, action: PlayerActions):
        if self.game_round == 0:
            if len(self.bet_sequences) >= 2 and self.bet_sequences[-1] == PlayerActions.CHECK_CALL:
                self.board_card = self._hidden_board_card
                self.game_round = 1
            else:
                self.bet_sequences[0] = self.bet_sequences[0] + (action,)
        if self.game_round == 1:
            self.bet_sequences[1] = self.bet_sequences[1] + (action,)

    def _get_half_pot(self):
        half_pot = 1  # Antes

        to_call = 0
        for action in self._bet_sequences[0]:
            if action == PlayerActions.FOLD:
                return half_pot
            elif action == PlayerActions.CHECK_CALL:
                half_pot += to_call
                to_call = 0
            elif action == PlayerActions.BET_RAISE:
                half_pot += to_call
                to_call = 2

        to_call = 0
        for action in self._bet_sequences[1]:
            if action == PlayerActions.FOLD:
                return half_pot
            elif action == PlayerActions.CHECK_CALL:
                half_pot += to_call
                to_call = 0
            elif action == PlayerActions.BET_RAISE:
                half_pot += to_call
                to_call = 4

        return half_pot

    def _get_winner(self, player_cards: List[int]):
        try:
            fold_idx = self._bet_sequences[0].index(PlayerActions.FOLD)
            unfolded_player = (fold_idx + 1) % 2
            return unfolded_player
        except ValueError:
            pass

        try:
            fold_idx = self._bet_sequences[1].index(PlayerActions.FOLD)
            unfolded_player = (fold_idx + 1) % 2
            return unfolded_player
        except ValueError:
            pass

        # Showdown
        if player_cards[0] == player_cards[1]:
            return None
        elif player_cards[0] == self.board_card:
            return 0
        elif player_cards[1] == self.board_card:
            return 1
        else:
            return 0 if player_cards[0] > player_cards[1] else 1

    def get_payoffs(self, player_cards: List[int]) -> np.ndarray:
        if not self.is_terminal:
            raise RuntimeError("Can't get payoffs for non-terminal")

        winner = self._get_winner(player_cards)
        if winner is None:
            return np.array([0, 0])

        half_pot = self._get_half_pot()

        if winner == 0:
            return np.array([half_pot, -half_pot])
        elif winner == 1:
            return np.array([-half_pot, half_pot])


class LeducInfoset(LeducNode):
    def __init__(self, card: int, bet_sequences: List[Tuple], board_card: Optional[int] = None):
        super().__init__(bet_sequences=bet_sequences, board_card=board_card)
        self.card = card

    def __eq__(self, other):
        return (self.card == other.card and self._bet_sequences == other.bet_sequences
                and self.board_card == other.board_card)


class LeducGameState(LeducNode):
    def __init__(
            self,
            player_cards: List[int],
            bet_sequences: List[Tuple],
            hidden_board_card: Optional[int],
            board_card: Optional[int] = None):
        self.player_cards = player_cards
        super().__init__(bet_sequences=bet_sequences, hidden_board_card=hidden_board_card, board_card=board_card)
        self.infosets = tuple(LeducInfoset(card, self._bet_sequences) for card in self.player_cards)

    def get_payoffs(self):
        return LeducNode.get_payoffs(self, self.player_cards)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "bet_sequences":
            self.infosets = tuple(LeducInfoset(card, self._bet_sequences) for card in self.player_cards)


class KuhnPokerGame(object):
    def __init__(self, player_cards: Optional[List[int]] = None):
        if player_cards is None:
            cards = random.sample(list(range(3)) + list(range(3)), 3)
            self.player_cards = cards[:2]
            self.hidden_board_card = cards[-1]   # board card hidden to start

        self.game_state = LeducGameState(player_cards, [(), ()], hidden_board_card=self.hidden_board_card)
