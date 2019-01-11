from KuhnPoker.NFSP.Dqn import QPolicy, KuhnQPolicy
from KuhnPoker.PolicyWrapper import NnPolicyWrapper, infoset_to_state
from KuhnPoker.Policies import Policy
from KuhnPoker.NFSP.Supervised import SupervisedNetwork, SupervisedTrainer
import random


class NfspAgent(Policy):
    def __init__(self, q_policy: QPolicy, supervised_trainer: SupervisedTrainer, nu: float):
        self.q_policy = q_policy
        self.supervised_trainer = supervised_trainer

        self.kuhn_rl_policy = KuhnQPolicy(self.q_policy)
        self.kuhn_supervised_policy = NnPolicyWrapper(self.supervised_trainer.network)

        self.nu = nu

        self.last_state = None
        self.last_action = None
        self.current_action = None
        self.current_state = None
        self.reward = None

    def reset(self):
        self.last_state = None
        self.last_action = None

    def aggressive_action_prob(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset):
        state = infoset_to_state(infoset)

        use_q = random.random() < self.nu
        if use_q:
            retval = self.kuhn_rl_policy.aggressive_action_prob(infoset)
            self.supervised_trainer.add_observation(state, retval)
        else:
            retval = self.kuhn_supervised_policy.aggressive_action_prob(infoset)

        self.last_state = self.current_state
        self.last_action = self.current_action
        self.current_action = retval
        self.current_state = state

        return retval

    def get_action(self, infoset: KuhnPoker.KuhnPokerGame.KuhnInfoset):
        retval = super().get_action(infoset)
        self.last_action = self.current_action
        self.current_action = retval

        return retval

    def notify_reward(self, reward: double, is_terminal: bool):
        self.reward = reward

        self.q_policy.step(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=self.current_state,
            is_terminal=is_terminal)
