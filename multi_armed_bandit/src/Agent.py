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

class RandomAgent(Agent):
    """ Random Agent in the MultiArmedBandit environment

        Takes action randomly
    """
    def get_action(self) -> ActType:
        return self.rng.integers(0, self.num_arms)

    def observe(self, action: ActType, reward: float, info: dict) -> int:
        return 0

class EpsGreedyAgent(Agent):
    """ The action-value method for this agent is epsilon-greedy i.e 
            the agent chooses a random action with probability epsilon and 
            chooses greedy action the rest of the time
    """
    def __init__(self, num_arms: int, seed: int, epsilon: float, optimism: float):
        self.epsilon = epsilon
        self.optimism = optimism
        super().__init__(num_arms, seed)
        
    def get_action(self) -> ActType:
        if self.rng.random() < epsilon:
            return self.rng.integers(0, self.num_arms)
        else:
            return int(np.argmax(self.Q))

    def observe(self, action: ActType, reward: float, info: dict):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

    def reset(self, seed: int):
        super().reset(seed=seed)
        self.Q = np.fill(self.num_arms, self.optimism)
        self.N = np.zeros(self.num_arms)
        