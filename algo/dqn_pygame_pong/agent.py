"""
This module contains the training agent which utilise neural network and replay memory
"""
import random
import math
import numpy as np
from algo.dqn_pygame_pong.dqn import DQN
from algo.dqn_pygame_pong.replay_memory import ReplayMemory


REPLAY_MEMORY_SIZE = 2000
REPLAY_BATCH_SIZE = 128
MEMORISE_DURATION = 750

GAMMA = 0.95

EPSILON_START = 1
EPSILON_MIN = 0.05
EPSILON_DECAY_RATE = 0.0005


class Agent:
    """
    This class uses a Deep Q-Network (DQN) agent for reinforcement learning in a given environment.
    The agent uses a neural network to estimate the Q-value function and employs experience replay and a
    target network to improve training stability.
    """
    def __init__(self, _num_state, _num_action):
        self.num_state = _num_state
        self.num_action = _num_action

        self.net = DQN(_num_state, _num_action)
        self.experience_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.observation_idx = 0
        self.epsilon = EPSILON_START

    def select_action(self, state):
        """
        Action selection based on epsilon greedy
        :param state:
        :return: action
        """
        if random.random() < self.epsilon or self.observation_idx < MEMORISE_DURATION:
            return random.randint(0, self.num_action - 1)
        return np.argmax(self.net._predict_single(state))

    def record_experience(self, experience):
        """
        Record an experience to memory
        :param experience:
        :return: None
        """
        self.experience_memory.memorise(experience)
        self.observation_idx += 1
        if self.observation_idx > MEMORISE_DURATION:
            self.epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * math.exp(
                -EPSILON_DECAY_RATE * (self.observation_idx - MEMORISE_DURATION)
            )

    def train(self):
        """
        Training algorithm
        :return:
        """
        batch = self.experience_memory.sample(REPLAY_BATCH_SIZE)
        _batch_size = len(batch)

        _state = np.zeros(self.num_state)

        current_state = np.array([item[0] for item in batch])
        target_state = np.array(
            [(_state if item[3] is None else item[3]) for item in batch]
        ) # 3 is the number of action

        policy_q = self.net._predict(current_state)
        target_q = self.net._predict(target_state)

        x = np.zeros((_batch_size, self.num_state))
        y = np.zeros((_batch_size, self.num_action))

        for i in range(_batch_size):
            batch_item = batch[i]
            state = batch_item[0]
            a = batch_item[1]
            reward = batch_item[2]
            next_state = batch_item[3]

            q_value = policy_q[i]
            if next_state is None:
                q_value[a] = reward
            else:
                q_value[a] = reward + GAMMA * np.amax(target_q[i])

            x[i] = state
            y[i] = q_value

        self.net._fit(x, y)
