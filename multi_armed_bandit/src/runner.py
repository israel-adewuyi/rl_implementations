import gym
import numpy as np

from environment import MultiArmedBandit
from Agent import RandomAgent
from utils import run_agent

max_episode_steps = 1000
num_arms = 10
stationary = True

if __name__ == '__main__':
    gym.envs.registration.register(
        id='KBandit/10ArmedBandit-v0',
        entry_point=MultiArmedBandit,
        max_episode_steps=max_episode_steps,
        nondeterministic=True,
        reward_threshold=1.0,
        kwargs={'num_arms':num_arms, 'stationary':stationary}
    )
    
    env = gym.make("KBandit/10ArmedBandit-v0")
    agent = RandomAgent(num_arms, 0)

    all_rewards, all_corrects = run_agent(env, agent)
    
    print(f"Expected correct freq: {1/10}, actual: {all_corrects.mean():.6f}")
    assert np.isclose(all_corrects.mean(), 1/10, atol=0.05), "Random agent is not random enough!"
    
    print(f"Expected average reward: 0.0, actual: {all_rewards.mean():.6f}")
    assert np.isclose(all_rewards.mean(), 0, atol=0.05), "Random agent should be getting mean arm reward, which is zero."