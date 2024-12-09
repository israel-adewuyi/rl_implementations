import gym
import numpy as np

from tqdm import tqdm
from Agent import Agent
from typing import Tuple
from gym.utils.step_api_compatibility import step_api_compatibility

def run_episode(env: gym.Env, agent: Agent, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    (rewards, best_action_taken) = ([], [])
    
    env.reset(seed=seed)
    agent.reset(seed=seed)
    
    done = False
    while not done:
        action = agent.get_action()
        (obs, reward, done, info) = step_api_compatibility(env.step(action), output_truncation_bool=False)
        agent.observe(action, reward, info)
        rewards.append(reward)
        best_action_taken.append(1 if action == info['best_arm'] else 0)
        # print(done)

    rewards = np.array(rewards, dtype=np.float64)
    best_action_taken = np.array(best_action_taken, dtype=np.int64)

    return (rewards, best_action_taken)
    
def run_agent(env: gym.Env, agent: Agent, n_runs: int = 200, base_seed: int = 1) -> None:
    all_rewards = []
    all_best_action_taken = []
    
    generator = np.random.default_rng(base_seed)
    for i in tqdm(range(n_runs)):
        seed = generator.integers(0, 10000).item()
        # print(f"Going on run {i} with seef {seed} of type {type(seed)}")
        rewards, best_actions = run_episode(env, agent, seed)
        # print(f"Finished the {i}-th run")
        all_rewards.append(rewards)
        all_best_action_taken.append(best_actions)

    all_rewards = np.array(all_rewards, dtype=np.float64)
    all_best_action_taken = np.array(all_best_action_taken, dtype=np.int64)

    return (all_rewards, all_best_action_taken)