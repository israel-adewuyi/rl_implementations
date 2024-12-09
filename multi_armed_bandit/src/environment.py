import gym
import numpy as np
from typing import Optional, Tuple

ObsType = int
ActType = int

max_episode_steps = 1000

class MultiArmedBandit(gym.Env):
    """
        This environment is a MultiArmedBandit environment (See Sutton and Barton)
        
    """
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    num_arms: int
    stationary: bool
    arm_reward_means: np.ndarray
    def __init__(self, num_arms: int, stationary: bool):
        # super().__init__()
        self.num_arms = num_arms
        self.stationary = stationary
        self.action_space = gym.spaces.Discrete(10)
        # No observation in this environment, so we just return 0 or any arbitrary number
        self.observation_space = gym.spaces.Discrete(1)
        self.reset()

    def step(self, arm: ActType) -> Tuple[ObsType, float, bool, dict]:
        assert self.arm_reward_means.contains(arm)

        if not self.stationary:
            drift = self.np_random.normal(loc=0.0, scale=0.0, size=(self.num_arms))
            self.arm_reward_means += drift
            self.best_arm = int(np.argmax(self.arm_reward_means))
        
        reward =  self.np_random.normal(loc=self.arm_reward_means[arm], scale=1.0)
        obs = 0
        done = False
        info = {'best_arm' : self.best_arm}
        return (obs, reward, done, info)
        

    def reset(self, seed: Optional[int] = None) -> ObsType:
        super().reset(seed=seed)

        if self.stationary:
            self.arm_reward_means = self.np_random.normal(loc=0.0, scale=1.0, size=(self.num_arms,))
        else:
            self.arm_reward_means = self.zeros(shape=(self.num_arms, ))
        
        print(self.arm_reward_means)

        self.best_arm = int(np.argmax(self.arm_reward_means))
        return 0

if __name__ == '__main__':
    gym.envs.registration.register(
        id='KBandit/10ArmedBandit-v0',
        entry_point=MultiArmedBandit,
        max_episode_steps=max_episode_steps,
        nondeterministic=True,
        reward_threshold=1.0,
        kwargs={'num_arms':10, 'stationary':True}
    )

    env = gym.make('KBandit/10ArmedBandit-v0')

    print(env)


