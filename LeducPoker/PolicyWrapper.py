import LeducPoker.Policies
import torch.nn as nn
import LeducPoker.LeducPokerGame
from LeducPoker.LeducPokerGame import PlayerActions
from Device import device
import torch
import numpy as np
from typing import Optional


def infoset_to_state(infoset: Optional[LeducPoker.LeducPokerGame.LeducInfoset]) -> np.ndarray:
    state = np.zeros(3 * 2 + 2 * 8)  # 22

    if infoset is None:
        return state

    state_idx = 0
    # 3 one-hots for card
    state[state_idx + infoset.card] = 1
    state_idx += 3

    # 3 one-hots for board
    if infoset.board_card:
        state[state_idx + infoset.board_card] = 1
    state_idx += 3

    # 2 rounds, 4 one-hot each (so 8 per round)
    # Rounds are encoded as one of [check/call; bet/raise]
    for action_idx, action in enumerate(infoset.bet_sequences[0]):
        if action == PlayerActions.CHECK_CALL:
            state[state_idx + 2 * action_idx] = 1
        else:
            state[state_idx + 2 * action_idx + 1] = 1
    state_idx += 2 * 4

    for action_idx, action in enumerate(infoset.bet_sequences[1]):
        if action == PlayerActions.CHECK_CALL:
            state[state_idx + 2 * action_idx] = 1
        else:
            state[state_idx + 2 * action_idx + 1] = 1

    return state


class NnPolicyWrapper(LeducPoker.Policies.Policy):
    def __init__(self, nn_policy: nn.Module):
        self.nn_policy = nn_policy

    def aggressive_action_prob(self, infoset: LeducPoker.LeducPokerGame.LeducInfoset):
        state = infoset_to_state(infoset)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        retval = self.nn_policy.forward(state)
        return retval.cpu().detach().numpy()[0]

