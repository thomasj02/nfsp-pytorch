from LeducPoker.NFSP.Dqn import QPolicy, LeducQPolicy
from LeducPoker.PolicyWrapper import infoset_to_state
from LeducPoker.Policies import Policy
from LeducPoker.NFSP.Supervised import SupervisedTrainer, SupervisedPolicy
from LeducPoker.LeducPokerGame import LeducInfoset, LeducPokerGame
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

    def reset(self):
        self.last_state = None
        self.last_action = None

    def action_prob(self, infoset: LeducInfoset):
        state = infoset_to_state(infoset)

        use_q = random.random() < self.nu
        if use_q:
            action = self.leduc_rl_policy.get_action(infoset)
            self.supervised_trainer.add_observation(state, action)
            retval = np.zeros(3)
            retval[action] = 1
        else:
            retval = self.leduc_supervised_policy.action_prob(infoset)

        self.last_state = state

        return retval

    def get_action(self, infoset: LeducInfoset):
        retval = super().get_action(infoset)
        # last_state set by aggressive_action_prob
        self.last_action = retval

        return retval

    def notify_reward(self, next_infoset: Optional[LeducInfoset], reward: float, is_terminal: bool):
        if self.last_action is None:
            assert reward == 0
            return

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


def collect_trajectories(agents: List[NfspAgent], num_games: int):
    with torch.no_grad():
        for agent in agents:
            agent.q_policy.qnetwork_target.eval()
            agent.q_policy.qnetwork_local.eval()
            agent.supervised_trainer.network.eval()

        for _ in range(num_games):
            game = LeducPokerGame(player_cards=None)

            for agent in agents:
                agent.reset()

            while not game.game_state.is_terminal:
                player_to_act = game.game_state.player_to_act
                if player_to_act == -1:
                    game.game_state.deal_board_card()
                    continue

                infoset = game.game_state.infosets[player_to_act]

                agent = agents[player_to_act]
                agent.notify_reward(next_infoset=infoset, reward=0, is_terminal=False)

                action = agent.get_action(infoset)

                game.game_state.add_action(action)
                if game.game_state.is_terminal:
                    game_rewards = game.game_state.get_payoffs()
                    for agent, reward in zip(agents, game_rewards):
                        agent.notify_reward(reward=reward, next_infoset=None, is_terminal=game.game_state.is_terminal)
