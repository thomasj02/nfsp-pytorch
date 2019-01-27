from typing import Tuple, Optional, List
import random
import numpy as np


class PlayerActions:
    BET_RAISE = 2
    CHECK_CALL = 1
    FOLD = 0

    ALL_ACTIONS = [FOLD, CHECK_CALL, BET_RAISE]

    ACTION_TO_CHAR = {
        FOLD: "f",
        CHECK_CALL: "c",
        BET_RAISE: "r"
    }


class LeducNode(object):
    def __init__(
            self,
            bet_sequences: List[Tuple[PlayerActions]],
            board_card: int):
        assert len(bet_sequences) == 2

        self._bet_sequences = bet_sequences
        if self.game_round == 0:
            self._hidden_board_card = board_card
            self.board_card = None
        else:
            self.board_card = board_card

    @property
    def game_round(self) -> int:
        return 1 if len(self.bet_sequences[0]) >= 2 and self.bet_sequences[0][-1] == PlayerActions.CHECK_CALL else 0

    @property
    def can_raise(self) -> bool:
        relevant_bet_sequences = self._relevant_bet_sequence()

        if len(relevant_bet_sequences) <= 1:
            return True
        else:
            return relevant_bet_sequences.count(PlayerActions.BET_RAISE) < 2

    @property
    def can_fold(self) -> bool:
        relevant_bet_sequence = self._relevant_bet_sequence()

        if len(relevant_bet_sequence) == 0:
            return False
        else:
            return relevant_bet_sequence[-1] == PlayerActions.BET_RAISE

    def _relevant_bet_sequence(self) -> Tuple[PlayerActions]:
        if self.game_round == 0:
            relevant_bet_sequence = self.bet_sequences[0]
        else:
            relevant_bet_sequence = self.bet_sequences[1]
        return relevant_bet_sequence

    @property
    def bet_sequences(self) -> List[Tuple[PlayerActions]]:
        return self._bet_sequences

    @property
    def is_terminal(self) -> bool:
        if len(self._bet_sequences[0]) > 0 and self._bet_sequences[0][-1] == PlayerActions.FOLD:
            return True

        if len(self._bet_sequences[1]) <= 1:
            return False

        if self._bet_sequences[1][-1] != PlayerActions.BET_RAISE:
            return True

        return False

    @property
    def player_to_act(self) -> int:
        relevant_bet_sequence = self._relevant_bet_sequence()
        return len(relevant_bet_sequence) % 2

    # Returns true if we transitioned to a new round, false otherwise
    def add_action(self, action: PlayerActions) -> bool:
        if action == PlayerActions.BET_RAISE:
            assert self.can_raise
        elif action == PlayerActions.FOLD:
            assert self.can_fold

        pre_action_game_round = self.game_round

        if self.game_round == 0:
            if len(self.bet_sequences) >= 2 and self.bet_sequences[-1] == PlayerActions.CHECK_CALL:
                self.board_card = self._hidden_board_card
            else:
                self.bet_sequences[0] = self.bet_sequences[0] + (action,)
        else:
            self.bet_sequences[1] = self.bet_sequences[1] + (action,)

        if pre_action_game_round == 0 and self.game_round == 1:
            self.board_card = self._hidden_board_card
            retval = True
        else:
            retval = False

        if self.game_round == 1:
            assert self.board_card is not None

        return retval

    def _get_half_pot(self) -> float:
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

    def _get_winner(self, player_cards: List[int]) -> Optional[int]:
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
        assert self.board_card is not None
        player_normalized_cards = [player_cards[0] % 3, player_cards[1] % 3]
        board_normalized_card = self.board_card % 3

        if player_normalized_cards[0] == player_normalized_cards[1]:
            return None
        elif player_normalized_cards[0] == board_normalized_card:
            return 0
        elif player_normalized_cards[1] == board_normalized_card:
            return 1
        else:
            return 0 if player_normalized_cards[0] > player_normalized_cards[1] else 1

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
    def __init__(
            self,
            card: int,
            bet_sequences: List[Tuple],
            board_card: int):
        super().__init__(bet_sequences=bet_sequences, board_card=board_card)
        self.card = card

    def __str__(self):
        card_to_char = {
            0: "J",
            1: "Q",
            2: "K"
        }
        retval = card_to_char[self.card % 3]
        if self.board_card:
            retval += card_to_char[self.board_card % 3]

        retval += ":/"
        retval += "".join(PlayerActions.ACTION_TO_CHAR[a] for a in self.bet_sequences[0])
        if self.game_round == 1:
            retval += "/"
            retval += "".join(PlayerActions.ACTION_TO_CHAR[a] for a in self.bet_sequences[1])
        retval += ":"
        return retval

    def __eq__(self, other):
        return (self.card == other.card and self._bet_sequences == other.bet_sequences
                and self.board_card == other.board_card)


class LeducGameState(LeducNode):
    def __init__(
            self,
            player_cards: List[int],
            bet_sequences: List[Tuple],
            board_card: int):
        self.player_cards = player_cards
        super().__init__(bet_sequences=bet_sequences, board_card=board_card)
        self.infosets = tuple(
            LeducInfoset(card, self._bet_sequences, board_card=board_card)
            for card in self.player_cards)

    def get_payoffs(self):
        return LeducNode.get_payoffs(self, self.player_cards)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "bet_sequences":
            self.infosets = tuple(
                LeducInfoset(card=card, bet_sequences=self._bet_sequences, board_card=self.board_card)
                for card in self.player_cards)


class LeducPokerGame(object):
    NUM_CARDS = 6
    DECK = tuple(range(6))

    def __init__(self, player_cards: Optional[List[int]] = None):
        if player_cards is None:
            cards = random.sample(self.DECK, 3)
            self.player_cards = cards[:2]
            self.hidden_board_card = cards[-1]   # board card hidden to start

        self.game_state = LeducGameState(player_cards, [(), ()], board_card=self.hidden_board_card)
