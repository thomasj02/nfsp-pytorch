import LeducPoker.LeducPokerGame
from LeducPoker.LeducPokerGame import PlayerActions
import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Policy(ABC):
    @abstractmethod
    def action_prob(self, infoset: LeducPoker.LeducPokerGame.LeducInfoset) -> List[float]:
        pass

    def get_action(self, infoset: LeducPoker.LeducPokerGame.LeducInfoset) -> int:
        act_prob = self.action_prob(infoset)
        retval = np.random.choice(
            [PlayerActions.FOLD, PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE], 1, p=act_prob)
        return retval


class AlwaysCheckCall(Policy):
    def action_prob(self, infoset: LeducPoker.LeducPokerGame.LeducInfoset) -> List[float]:
        retval = [0, 0, 0]
        retval[PlayerActions.CHECK_CALL] = 1
        return retval

    def get_action(self, infoset: LeducPoker.LeducPokerGame.LeducInfoset) -> int:
        return PlayerActions.CHECK_CALL


class AlwaysBet(Policy):
    def action_prob(self, infoset: LeducPoker.LeducPokerGame.LeducInfoset) -> List[float]:
        retval = [0, 0, 0]
        if infoset.can_raise:
            retval[PlayerActions.BET_RAISE] = 1
        else:
            retval[PlayerActions.CHECK_CALL] = 1
        return retval

    def get_action(self, infoset: LeducPoker.LeducPokerGame.LeducInfoset) -> int:
        return PlayerActions.BET_RAISE


class NashPolicy(Policy):
    def __init__(self, p0_strat_filename, p1_strat_filename):
        self.player_policies = [
            self.load_policy(p0_strat_filename),
            self.load_policy(p1_strat_filename)
        ]

    def load_policy(self, policy_filename):
        char_to_card = {'J': 0, 'Q': 1, 'K': 2}
        char_to_action = {
            'r': PlayerActions.BET_RAISE,
            'c': PlayerActions.CHECK_CALL
        }

        retval = {}
        with open(policy_filename, "r") as policy_filename:
            for line in policy_filename:
                if line[0] == "#":
                    continue

                cards, actions, probs = line.split(":")

                player_card = char_to_card[cards[0]]
                if len(cards) > 1:
                    board_card = char_to_card[cards[1]]
                else:
                    board_card = None

                round_actions = actions.split("/")[1:]
                round1_actions = tuple(char_to_action[c] for c in round_actions[0])
                if len(round_actions) > 1:
                    round2_actions = tuple(char_to_action[c] for c in round_actions[1])
                else:
                    round2_actions = ()

                infoset_template = (
                    player_card, board_card, round1_actions, round2_actions
                )

                raise_prob, call_prob, fold_prob = list([float(p) for p in probs.strip(" \n").split(" ")])
                infoset_probs = [0, 0, 0]
                infoset_probs[PlayerActions.BET_RAISE] = raise_prob
                infoset_probs[PlayerActions.CHECK_CALL] = call_prob
                infoset_probs[PlayerActions.FOLD] = fold_prob

                retval[infoset_template] = infoset_probs

        return retval

    def action_prob(self, infoset: LeducPoker.LeducPokerGame.LeducInfoset) -> List[float]:
        assert not infoset.is_terminal

        if infoset.board_card is not None:
            board_card = infoset.board_card % 3
        else:
            board_card = None

        infoset_template = (
            infoset.card % 3, board_card, infoset.bet_sequences[0], infoset.bet_sequences[1]
        )

        return self.player_policies[infoset.player_to_act][infoset_template]
