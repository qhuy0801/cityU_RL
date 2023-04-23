"""
This module contain memory for implementation of experience replay
"""
import collections
import random


class ReplayMemory:
    """
    A simple memory that automatically delete old records if reach capacity
    """
    def __init__(self, memory_size):
        self.memory = collections.deque(maxlen=memory_size)

    def memorise(self, sample):
        """
        :param sample:
        :return:
        """
        self.memory.append(sample)

    def sample(self, _batch_size):
        """
        :param _batch_size:
        :return:
        """
        batch_size = min(_batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)
