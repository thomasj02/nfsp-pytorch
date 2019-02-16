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
            board_card: Optional[int]):
        assert len(bet_sequences) == 2

        self._bet_sequences = bet_sequences
        self.board_card = board_card

        if self.game_round == 1:
            assert self.board_card is not None

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

    def fixup_action(self, action: PlayerActions):
        if action == PlayerActions.FOLD and not self.can_fold:
            return PlayerActions.CHECK_CALL
        elif action == PlayerActions.BET_RAISE and not self.can_raise:
            return PlayerActions.CHECK_CALL
        else:
            return action

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
        if self.game_round == 1 and self.board_card is None:
            return -1  # Chance
        relevant_bet_sequence = self._relevant_bet_sequence()
        return len(relevant_bet_sequence) % 2

    # Returns cost of taking action
    def add_action(self, action: PlayerActions) -> int:
        action = self.fixup_action(action)

        game_round = self.game_round
        retval = 0
        if game_round == 0:
            if len(self.bet_sequences[0]) < 2:
                retval = 1  # Antes

            if len(self.bet_sequences[0]) > 0 and self.bet_sequences[0][-1] == PlayerActions.BET_RAISE:
                retval += 2  # 2 to call

            if action == PlayerActions.BET_RAISE:
                retval += 2

            self.bet_sequences[0] = self.bet_sequences[0] + (action,)
        else:
            if len(self.bet_sequences[1]) > 0 and self.bet_sequences[1][-1] == PlayerActions.BET_RAISE:
                retval = 4  # 4 to call

            if action == PlayerActions.BET_RAISE:
                retval += 4

            self.bet_sequences[1] = self.bet_sequences[1] + (action,)

        if self.game_round == 1 and self.player_to_act != -1:
            assert self.board_card is not None
        else:
            assert self.board_card is None

        # one fixup: if they folded
        if action == PlayerActions.FOLD:
            if game_round == 0 and len(self.bet_sequences[0]) <= 2:
                retval = 1  # Ante
            else:
                retval = 0

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

        half_pot = self._get_half_pot()

        winner = self._get_winner(player_cards)
        if winner is None:
            return np.array([half_pot, half_pot])

        if winner == 0:
            return np.array([half_pot * 2, 0])
        elif winner == 1:
            return np.array([0, half_pot * 2])


class LeducInfoset(LeducNode):
    def __init__(
            self,
            card: int,
            bet_sequences: List[Tuple],
            board_card: Optional[int]):
        super().__init__(bet_sequences=bet_sequences, board_card=board_card)
        self.card = card

    def __str__(self):
        card_to_char = {
            0: "J",
            1: "Q",
            2: "K"
        }
        retval = card_to_char[self.card % 3]
        if self.board_card is not None:
            retval += card_to_char[self.board_card % 3]

        retval += ":/"
        retval += "".join(PlayerActions.ACTION_TO_CHAR[a] for a in self.bet_sequences[0])
        if self.game_round == 1:
            retval += "/"
            retval += "".join(PlayerActions.ACTION_TO_CHAR[a] for a in self.bet_sequences[1])
        retval += ":"
        return retval

    def __eq__(self, other):
        if other is None:
            return False
        return (self.card == other.card and self._bet_sequences == other.bet_sequences
                and self.board_card == other.board_card)


class LeducGameState(LeducNode):
    def __init__(
            self,
            player_cards: List[int],
            bet_sequences: List[Tuple],
            board_card: Optional[int]):
        self.player_cards = player_cards
        super().__init__(bet_sequences=bet_sequences, board_card=board_card)
        self.infosets = tuple(
            LeducInfoset(card, self._bet_sequences, board_card=board_card)
            for card in self.player_cards)

    def _update_infosets(self):
        self.infosets = tuple(
            LeducInfoset(card=card, bet_sequences=self._bet_sequences, board_card=self.board_card) for card in
            self.player_cards)

    def deal_board_card(self):
        assert self.board_card is None and self.player_to_act == -1
        deck = list(LeducPokerGame.DECK)
        deck.remove(self.player_cards[0])
        deck.remove(self.player_cards[1])
        self.board_card = random.choice(deck)
        self._update_infosets()

    def get_payoffs(self):
        return LeducNode.get_payoffs(self, self.player_cards)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == "bet_sequences":
            self._update_infosets()


class LeducPokerGame(object):
    NUM_CARDS = 6
    DECK = tuple(range(6))

    def __init__(self, player_cards: Optional[List[int]] = None):
        if player_cards is None:
            cards = random.sample(self.DECK, 2)
            self.player_cards = cards

        self.game_state = LeducGameState(self.player_cards, [(), ()], board_card=None)
