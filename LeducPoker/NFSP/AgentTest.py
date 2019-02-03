import unittest
import LeducPoker.NFSP.Agent
from unittest.mock import Mock, MagicMock
from unittest import mock
from LeducPoker.LeducPokerGame import LeducInfoset
from LeducPoker.PolicyWrapper import infoset_to_state
from LeducPoker.LeducPokerGame import PlayerActions
from mock import call
import copy

class CopyingMagicMock(mock.MagicMock):
    def _mock_call(_mock_self, *args, **kwargs):
        return super(CopyingMagicMock, _mock_self)._mock_call(*copy.deepcopy(args), **copy.deepcopy(kwargs))


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.mock_q_policy = Mock()
        self.mock_supervised_trainer = Mock()

        self.sut = LeducPoker.NFSP.Agent.NfspAgent(self.mock_q_policy, self.mock_supervised_trainer, nu=0.1)

    def test_action_prob_q(self):
        self.sut = LeducPoker.NFSP.Agent.NfspAgent(self.mock_q_policy, self.mock_supervised_trainer, nu=1.1)

        self.sut.leduc_rl_policy.get_action = MagicMock(return_value=1)
        self.sut.supervised_trainer.add_observation = MagicMock()

        infoset = LeducInfoset(card=1, bet_sequences=[(PlayerActions.CHECK_CALL,), ()], board_card=None)
        infoset_state = infoset_to_state(infoset)

        retval = self.sut.action_prob(infoset)

        self.assertEqual(1, retval)
        self.assertEqual(infoset_state.tolist(), self.sut.last_state.tolist())

        self.sut.leduc_rl_policy.get_action.assert_called_with(infoset)

        self.assertEqual(self.sut.supervised_trainer.add_observation.call_args[0][0].tolist(), infoset_state.tolist())
        self.assertEqual(self.sut.supervised_trainer.add_observation.call_args[0][1], 1)

    def test_action_prob_supervised(self):
        self.sut = LeducPoker.NFSP.Agent.NfspAgent(self.mock_q_policy, self.mock_supervised_trainer, nu=0)

        self.sut.leduc_supervised_policy.action_prob = MagicMock(return_value=[1, 0, 0])
        infoset = LeducInfoset(card=1, bet_sequences=[(PlayerActions.BET_RAISE,), ()], board_card=None)

        retval = self.sut.action_prob(infoset)

        self.assertEqual([1, 0, 0], retval)
        self.sut.leduc_supervised_policy.action_prob.assert_called_with(infoset)

    def test_notify_reward(self):
        self.sut = LeducPoker.NFSP.Agent.NfspAgent(self.mock_q_policy, self.mock_supervised_trainer, nu=0)
        self.sut.leduc_supervised_policy.action_prob = MagicMock(return_value=[0, 1, 0])

        infoset = LeducInfoset(card=1, bet_sequences=[(PlayerActions.CHECK_CALL,), ()], board_card=None)
        infoset_state = infoset_to_state(infoset)
        self.sut.get_action(infoset)

        self.mock_q_policy.add_sars = MagicMock()

        infoset_next = LeducInfoset(card=1, bet_sequences=[(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE), ()], board_card=None)
        infoset_next_state = infoset_to_state(infoset_next)

        self.sut.notify_reward(next_infoset=infoset_next, reward=123, is_terminal=True)

        # call_args[0] are the position args
        self.assertEqual(self.mock_q_policy.add_sars.call_args[0], tuple())
        self.assertEqual(self.mock_q_policy.add_sars.call_args[1]["state"].tolist(), infoset_state.tolist())
        self.assertEqual(self.mock_q_policy.add_sars.call_args[1]["action"], PlayerActions.CHECK_CALL)
        self.assertEqual(self.mock_q_policy.add_sars.call_args[1]["reward"], 123)
        self.assertEqual(self.mock_q_policy.add_sars.call_args[1]["next_state"].tolist(), infoset_next_state.tolist())
        self.assertEqual(self.mock_q_policy.add_sars.call_args[1]["is_terminal"], True)


class CollectTrajectoriesTest(unittest.TestCase):
    def setUp(self):
        self.agents = [MagicMock(), MagicMock()]
        for agent in self.agents:
            agent.reset = MagicMock()
            agent.notify_reward = CopyingMagicMock()
            agent.q_policy = MagicMock()
            agent.q_policy.qnetwork_local = MagicMock()
            agent.q_policy.qnetwork_target = MagicMock()
            agent.q_policy.supervised_trainer = MagicMock()

    def test_bet_fold_game(self):
        def mock_random_sample(a, b):
            return [1, 0]

        def get_agent0_action(infoset: LeducInfoset):
            return PlayerActions.BET_RAISE

        def get_agent1_action(infoset: LeducInfoset):
            return PlayerActions.FOLD

        # P0 has queen, P1 has jack
        with mock.patch('random.sample', mock_random_sample):
            self.agents[0].get_action = MagicMock(side_effect=get_agent0_action)
            self.agents[1].get_action = MagicMock(side_effect=get_agent1_action)
            LeducPoker.NFSP.Agent.collect_trajectories(self.agents, num_games=1)

        self.agents[0].reset.assert_called_once_with()
        self.agents[1].reset.assert_called_once_with()

        self.assertEqual(
            self.agents[0].notify_reward.mock_calls[0][2],
            {"next_infoset": LeducInfoset(card=1, bet_sequences=[(),()], board_card=None), "reward": 0, "is_terminal": False})
        self.assertEqual(
            self.agents[1].notify_reward.mock_calls[0][2],
            {"next_infoset": LeducInfoset(card=0, bet_sequences=[(PlayerActions.BET_RAISE,), ()], board_card=None),
             "reward": 0, "is_terminal": False})

        self.assertEqual(
            self.agents[0].notify_reward.mock_calls[1][2],
            {"next_infoset": None, "reward": 1, "is_terminal": True})
        self.assertEqual(
            self.agents[1].notify_reward.mock_calls[1][2],
            {"next_infoset": None, "reward": -1, "is_terminal": True})

        self.assertEqual(2, len(self.agents[0].notify_reward.mock_calls))
        self.assertEqual(2, len(self.agents[1].notify_reward.mock_calls))

    def test_check_bet_call_game(self):
        def mock_random_sample(a, b):
            return [1, 2]

        def mock_random_choice(a):
            return 2

        def get_agent0_action(infoset: LeducInfoset):
            return PlayerActions.CHECK_CALL

        def get_agent1_action(infoset: LeducInfoset):
            return PlayerActions.BET_RAISE

        # P0 has queen, P1 has king
        with mock.patch('random.sample', mock_random_sample):
            with mock.patch('random.choice', mock_random_choice):
                self.agents[0].get_action = MagicMock(side_effect=get_agent0_action)
                self.agents[1].get_action = MagicMock(side_effect=get_agent1_action)
                LeducPoker.NFSP.Agent.collect_trajectories(self.agents, num_games=1)

        self.agents[0].reset.assert_called_once_with()
        self.agents[1].reset.assert_called_once_with()

        self.assertEqual(
            self.agents[0].notify_reward.mock_calls[0][2],
            {"next_infoset": LeducInfoset(1, bet_sequences=[(), ()], board_card=None), "reward": 0, "is_terminal": False})
        self.assertEqual(
            self.agents[1].notify_reward.mock_calls[0][2],
            {"next_infoset": LeducInfoset(2, bet_sequences=[(PlayerActions.CHECK_CALL,), ()], board_card=None),
             "reward": 0, "is_terminal": False})
        self.assertEqual(
            self.agents[0].notify_reward.mock_calls[1][2],
            {"next_infoset": LeducInfoset(1, bet_sequences=[(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE), ()],
                                          board_card=None),
             "reward": 0, "is_terminal": False})

        # 2nd round
        self.assertEqual(
            self.agents[0].notify_reward.mock_calls[2][2],
            {"next_infoset": LeducInfoset(1, bet_sequences=[(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL), ()], board_card=2),
             "reward": 0, "is_terminal": False})
        self.assertEqual(
            self.agents[1].notify_reward.mock_calls[1][2],
            {"next_infoset": LeducInfoset(2, bet_sequences=[(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL), (PlayerActions.CHECK_CALL,)], board_card=2),
             "reward": 0, "is_terminal": False})
        self.assertEqual(
            self.agents[0].notify_reward.mock_calls[3][2],
            {"next_infoset": LeducInfoset(1, bet_sequences=[(PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE, PlayerActions.CHECK_CALL), (PlayerActions.CHECK_CALL, PlayerActions.BET_RAISE)],
                                          board_card=2),
             "reward": 0, "is_terminal": False})

        # Terminals
        self.assertEqual(
            self.agents[1].notify_reward.mock_calls[2][2],
            {"next_infoset": None, "reward": 7, "is_terminal": True})
        self.assertEqual(
            self.agents[0].notify_reward.mock_calls[4][2],
            {"next_infoset": None, "reward": -7, "is_terminal": True})

        self.assertEqual(5, len(self.agents[0].notify_reward.mock_calls))
        self.assertEqual(3, len(self.agents[1].notify_reward.mock_calls))
