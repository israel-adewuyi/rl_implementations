import gym
import numpy as np

ObsType = int
ActType = int

class Agent:
    """ Base Agent class for the MultiArmedBandit Environment"""

    rng = np.random.default_rng()

    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)

    def get_action(self) -> ActType:
        raise NotImplementedError

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        raise NotImplementedError

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        