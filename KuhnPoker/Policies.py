import KuhnPoker.KuhnPoker
import random
from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPoker.KuhnInfoset) -> float:
        pass

    def get_action(self, infoset: KuhnPoker.KuhnPoker.KuhnInfoset) -> int:
        agg_act_prob = self.aggressive_action_prob(infoset)
        rand_num = random.random()
        if rand_num < agg_act_prob:
            return 1
        else:
            return 0


class AlwaysCheckCall(Policy):
    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPoker.KuhnInfoset) -> float:
        return 0

    def get_action(self, infoset: KuhnPoker.KuhnPoker.KuhnInfoset) -> int:
        return 0


class AlwaysBet(Policy):
    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPoker.KuhnInfoset) -> float:
        return 1

    def get_action(self, infoset: KuhnPoker.KuhnPoker.KuhnInfoset) -> int:
        return 1


class RandomPolicy(Policy):
    def __init__(self, aggressive_action_prob):
        self.agg_act_prob = aggressive_action_prob

    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPoker.KuhnInfoset):
        return self.agg_act_prob
