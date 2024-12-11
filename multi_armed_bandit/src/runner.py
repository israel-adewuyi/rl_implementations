import gym
import numpy as np

from environment import MultiArmedBandit
from Agent import RandomAgent, EpsGreedyAgent, CheatyMcCheater
from utils import run_agent, plot_rewards

max_episode_steps = 1000
num_arms = 10
stationary = True
epsilon = 0.01
optimism = 0

if __name__ == '__main__':
    gym.envs.registration.register(
        id='KBandit/10ArmedBandit-v0',
        entry_point=MultiArmedBandit,
        max_episode_steps=max_episode_steps,
        nondeterministic=True,
        reward_threshold=1.0,
        kwargs={'num_arms':num_arms, 'stationary':stationary}
    )
    
    env = gym.make("KBandit/10ArmedBandit-v0", stationary=True)
    agent = RandomAgent(num_arms, 0)

    all_rewards, names = [], []

    rewards, correct = run_agent(env, agent, 200, 42)
    name = f"random_agent"
    all_rewards.append(rewards)
    names.append(name)
    print(name)
    print(f" -> Frequency of correct arm: {correct.mean():.4f}")
    print(f" -> Average reward: {rewards.mean():.4f}")

    for optimism in [0, 5]:
        agent = EpsGreedyAgent(num_arms=num_arms, seed=0, epsilon=epsilon, optimism=optimism)
        name = f"eps_greedy_{epsilon}_{optimism}"
        rewards, correct = run_agent(env, agent, n_runs=200, base_seed=42)
        all_rewards.append(rewards)
        names.append(name)
        print(name)
        print(f" -> Frequency of correct arm: {correct.mean():.4f}")
        print(f" -> Average reward: {rewards.mean():.4f}")

    cheat_agent = CheatyMcCheater(num_arms, 0)
    rewards, correct = run_agent(env, cheat_agent, 200, 42)
    name = f"cheating_agent"
    all_rewards.append(rewards)
    names.append(name)
    print(name)
    print(f" -> Frequency of correct arm: {correct.mean():.4f}")
    print(f" -> Average reward: {rewards.mean():.4f}")

    plot_rewards(all_rewards, names, 15)
        
