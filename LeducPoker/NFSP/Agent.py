from LeducPoker.NFSP.Dqn import QPolicy, LeducQPolicy
from LeducPoker.PolicyWrapper import infoset_to_state
from LeducPoker.Policies import Policy
from LeducPoker.NFSP.Supervised import SupervisedTrainer, SupervisedPolicy
from LeducPoker.LeducPokerGame import LeducInfoset, LeducPokerGame, PlayerActions
from typing import List, Optional
import random
import torch
import torch.nn
import numpy as np


class NfspAgent(Policy):
    def __init__(self, q_policy: QPolicy, supervised_trainer: SupervisedTrainer, nu: float):
        self.q_policy = q_policy
        self.supervised_trainer = supervised_trainer

        self.leduc_rl_policy = LeducQPolicy(self.q_policy)
        self.leduc_supervised_policy = SupervisedPolicy(self.supervised_trainer.network)

        self.nu = nu

        self.last_state = None
        self.last_action = None
        self.use_q_policy = None

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.use_q_policy = random.random() < self.nu

    def action_prob(self, infoset: LeducInfoset):
        state = infoset_to_state(infoset)

        if self.use_q_policy:
            retval = self.leduc_rl_policy.action_prob(infoset)
        else:
            retval = self.leduc_supervised_policy.action_prob(infoset)

        self.last_state = state

        return retval

    def get_action(self, infoset: LeducInfoset):
        retval = super().get_action(infoset)
        retval = retval[0]

        if self.use_q_policy:
            self.supervised_trainer.add_observation(self.last_state, retval)

        # last_state set by aggressive_action_prob
        self.last_action = retval

        return retval

    def notify_reward(self, next_infoset: Optional[LeducInfoset], reward: float, is_terminal: bool):
        if self.last_action is None:
            assert reward == 0
            return False

        if next_infoset is None:
            assert is_terminal

        assert self.last_state is not None
        assert self.last_action is not None

        next_state = infoset_to_state(next_infoset)
        self.q_policy.add_sars(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=next_state,
            is_terminal=is_terminal)
        return True


class CompositePolicy(Policy):
    def __init__(self, policies: List[Policy]):
        self._policies = policies

    def _get_policy(self, infoset: LeducInfoset):
        player_to_act = infoset.player_to_act
        assert 0 <= player_to_act < len(self._policies)
        policy = self._policies[player_to_act]
        return policy

    def action_prob(self, infoset: LeducInfoset):
        policy = self._get_policy(infoset)
        return policy.action_prob(infoset)

    def get_action(self, infoset: LeducInfoset):
        policy = self._get_policy(infoset)
        return policy.get_action(infoset)


class TrajectoriesCollecter:
    def __init__(self, agents: List[NfspAgent]):
        self.game = None
        self.agents = agents

        self.cum_rewards = None
        self.player_rewards = None

        self.reset_game()

    def reset_game(self):
        self.game = LeducPokerGame(player_cards=None)
        for agent in self.agents:
            agent.reset()

        self.cum_rewards = np.zeros(2, dtype=float)  # Verification
        self.player_rewards = np.zeros(2, dtype=float)

    def collect_trajectories(self, max_samples: Optional[int]):
        samples_collected = 0
        with torch.no_grad():
            for agent in self.agents:
                agent.q_policy.eval()
                agent.supervised_trainer.network.eval()

            while samples_collected < max_samples:
                player_to_act = self.game.game_state.player_to_act

                infoset = self.game.game_state.infosets[player_to_act]

                agent = self.agents[player_to_act]
                agent.notify_reward(
                        next_infoset=infoset,
                        reward=self.player_rewards[player_to_act],
                        is_terminal=False)
                samples_collected += 1

                self.cum_rewards[player_to_act] += self.player_rewards[player_to_act]

                action = agent.get_action(infoset)

                action_cost, action = self.game.game_state.add_action(action)
                self.player_rewards[player_to_act] = -action_cost

                if self.game.game_state.is_terminal:
                    # This is so hacky
                    # if action == PlayerActions.FOLD:
                    #     # Return the bet
                    #     if self.game.game_state.game_round == 0:
                    #         self.player_rewards[(player_to_act + 1) % 2] += 2
                    #     else:
                    #         self.player_rewards[(player_to_act + 1) % 2] += 4

                    game_rewards = self.game.game_state.get_payoffs()
                    game_rewards += self.player_rewards
                    self.cum_rewards += game_rewards
                    #assert(sum(self.cum_rewards) == 2)  # The 2 antes magically appear in lua code

                    for agent, reward in zip(self.agents, game_rewards):
                        agent.notify_reward(
                            reward=reward, next_infoset=None, is_terminal=self.game.game_state.is_terminal)

                    self.reset_game()

            return samples_collected
