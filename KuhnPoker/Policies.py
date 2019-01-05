import KuhnPoker.KuhnPokerGame
import random
from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset) -> float:
        pass

    def get_action(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset) -> int:
        agg_act_prob = self.aggressive_action_prob(infoset)
        rand_num = random.random()
        if rand_num < agg_act_prob:
            return 1
        else:
            return 0


class AlwaysCheckCall(Policy):
    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset) -> float:
        return 0

    def get_action(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset) -> int:
        return 0


class AlwaysBet(Policy):
    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset) -> float:
        return 1

    def get_action(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset) -> int:
        return 1


class RandomPolicy(Policy):
    def __init__(self, aggressive_action_prob):
        self.agg_act_prob = aggressive_action_prob

    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset):
        return self.agg_act_prob


class NashPolicy(Policy):
    def __init__(self, alpha):
        assert 0 <= alpha <= 1.0 / 3.0
        self.alpha = alpha

    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset) -> float:
        assert not infoset.is_terminal

        if infoset.player_to_act == 0:
            if len(infoset.bet_sequence) == 0:
                if infoset.card == 0:
                    return self.alpha
                elif infoset.card == 1:
                    return 0
                else:
                    return 3.0 * self.alpha
            else:  # p1 bet
                if infoset.card == 0:
                    return 0
                elif infoset.card == 1:
                    return self.alpha + 1.0 / 3.0
                else:
                    return 1
        else:
            if infoset.bet_sequence == (0,):
                if infoset.card == 0:
                    return 1.0 / 3.0
                elif infoset.card == 1:
                    return 0
                else:
                    return 1.0
            elif infoset.bet_sequence == (1,):
                if infoset.card == 0:
                    return 0
                elif infoset.card == 1:
                    return 1.0 / 3.0
                else:
                    return 1
