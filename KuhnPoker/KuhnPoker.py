from typing import Tuple, Optional
import random


class KuhnNode(object):
    def __init__(self, bet_sequence):
        self.bet_sequence = bet_sequence

    @property
    def is_terminal(self):
        if len(self.bet_sequence) <= 1:
            return False
        elif len(self.bet_sequence) == 3:
            return True
        elif self.bet_sequence[0] == 0 and self.bet_sequence[1] == 1:
            return False
        else:
            return True

    @property
    def player_to_act(self):
        return len(self.bet_sequence) % 2

    @staticmethod
    def _p0_showdown_payoff(player_cards: Tuple[int, int]) -> float:
        return int(player_cards[0] > player_cards[1]) * 2 - 1

    def get_payoffs(self, player_cards: Tuple[int, int]) -> Tuple[float, float]:
        if not self.is_terminal:
            raise RuntimeError("Can't get payoffs for non-terminal")

        if self.bet_sequence == (0, 0):
            p0_payoff = self._p0_showdown_payoff(player_cards)
        elif self.bet_sequence == (0, 1, 0):
            p0_payoff = -1
        elif self.bet_sequence == (0, 1, 1):
            p0_payoff = self._p0_showdown_payoff(player_cards) * 2
        elif self.bet_sequence == (0, 1, 0):
            p0_payoff = -1
        elif self.bet_sequence == (1, 0):
            p0_payoff = 1
        else:
            p0_payoff = self._p0_showdown_payoff(player_cards) * 2

        return p0_payoff, -p0_payoff


class KuhnInfoset(KuhnNode):
    def __init__(self, card: int, bet_sequence: Tuple[int, ...]):
        super().__init__(bet_sequence)
        self.card = card


class KuhnGameState(KuhnNode):
    def __init__(self, player_cards: Tuple[int, int], bet_sequence: Tuple[int, ...]):
        super().__init__(bet_sequence)
        self.player_cards = player_cards
        self.infosets = (KuhnInfoset(card, self.bet_sequence) for card in self.player_cards)


class KuhnPokerGame(object):
    def __init__(self, player_cards: Optional[Tuple[int, int]] = None):
        if player_cards is None:
            player_cards = random.sample(range(3), 2)

        self.game_state = KuhnGameState(player_cards, ())
