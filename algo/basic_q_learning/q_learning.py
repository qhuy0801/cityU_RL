import numpy as np
import random
from IPython.display import clear_output
from algo.e_greedy.epsilon_greedy import get_epsilon

import pandas as pd

def q_learning(
    env,
    penalty,
    max_eps,
    alpha,
    gamma,
    epsilon_start,
    strategy="linear",
    epsilon_decay=None,
):
    # initialise the q_table
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # initialise training information
    num_step_info = []
    num_penalty_info = []
    total_reward_info = []
    epsilon_info = []

    for episode in range(1, max_eps + 1):
        # reset the environment at the beginning of episode
        state = env.reset()

        # initialise training info of each episode
        (
            step_count,
            penalty_count,
            total_reward,
        ) = (
            0,
            0,
            0,
        )
        termination = False

        # get epsilon of the episode
        epsilon = get_epsilon(epsilon_start, max_eps, episode, strategy, epsilon_decay)

        # keep updating until get termination signal
        while not termination:
            # action selection based on epsilon
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values

            # gather new state from action
            new_state, reward, termination, _ = env.step(action)

            # obtain current state so that we won't lose it
            current_q = q_table[state, action]

            # maximum expected future rewards
            max_expected = np.max(q_table[new_state])

            # calculate Q-values and fill it
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_expected)
            q_table[state, action] = new_q

            # gather training info:
            # number of steps to complete the episode
            step_count += 1

            # number of penalties received
            if reward == penalty:
                penalty_count += 1

            # accumulated rewards
            total_reward += reward

            # assign the new state
            state = new_state

        num_step_info.append(step_count)
        num_penalty_info.append(penalty_count)
        total_reward_info.append(total_reward)
        epsilon_info.append(epsilon)

        if episode % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {episode}")

    print("Training finished.\n")

    # gather training information
    training_info = pd.DataFrame(
        {
            "episode": range(1, max_eps + 1),
            "num_steps": num_step_info,
            "num_penalties": num_penalty_info,
            "total_rewards": total_reward_info,
            "epsilon": epsilon_info,
        }
    )

    return q_table, training_info