import gym
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm
from Agent import Agent
from typing import Tuple, List
from gym.utils.step_api_compatibility import step_api_compatibility

def run_episode(env: gym.Env, agent: Agent, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rewards, best_action_taken = [], []
    
    env.reset(seed=seed)
    agent.reset(seed=seed)
    
    done = False
    while not done:
        action = agent.get_action()
        (obs, reward, done, info) = step_api_compatibility(env.step(action), output_truncation_bool=False)
        agent.observe(action, reward, info)
        rewards.append(reward)
        best_action_taken.append(1 if action == info['best_arm'] else 0)

    rewards = np.array(rewards, dtype=np.float64)
    best_action_taken = np.array(best_action_taken, dtype=np.int64)

    return (rewards, best_action_taken)
    
def run_agent(env: gym.Env, agent: Agent, n_runs: int = 200, base_seed: int = 1) -> None:
    all_rewards = []
    all_best_action_taken = []
    
    generator = np.random.default_rng(base_seed)
    for i in tqdm(range(n_runs)):
        seed = generator.integers(low=0, high=10000, size=1).item()
        rewards, best_actions = run_episode(env, agent, seed)
        all_rewards.append(rewards)
        all_best_action_taken.append(best_actions)

    all_rewards = np.array(all_rewards, dtype=np.float64)
    all_best_action_taken = np.array(all_best_action_taken, dtype=np.int64)

    return (all_rewards, all_best_action_taken)

def moving_avg(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_rewards(
    all_rewards: List[np.ndarray],
    names: List[str],
    moving_avg_window: int = 15,
):
    fig = go.Figure(layout=dict(template="simple_white", title_text="Mean reward over all runs"))
    for rewards, name in zip(all_rewards, names):
        rewards_avg = rewards.mean(axis=0)
        if moving_avg_window is not None:
            rewards_avg = moving_avg(rewards_avg, moving_avg_window)
        fig.add_trace(go.Scatter(y=rewards_avg, mode="lines", name=name))
    # fig.show()
    fig.write_image('artefacts/random1.png')