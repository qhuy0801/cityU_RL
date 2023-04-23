"""
This section has the primary training loop for a DQN agent in the Pong game.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algo.dqn_pygame_pong import agent
from envs import pong_env
from helpers.visualising_helper import plot_training_pong

# environment definition
ACTION_COUNT = 3 # go up, down or stay
STATE_COUNT = 5

# maximum frame (observation) to stop the train if average score not archived
MAXIMUM_FRAME_COUNT = 20000

SCREEN_SIZE = (400, 400)


def normalise_state(
    _y_our_paddle, _x_ball, _y_ball, _x_ball_direction, _y_ball_direction
):
    """
    Normalizes the given list of factors and return as the same order.
    :param _y_our_paddle: position of our agent paddle
    :param _x_ball: horizontal coordinate of ball
    :param _y_ball: vertical coordinate of ball
    :param _x_ball_direction: horizontal moving vector of ball
    :param _y_ball_direction: vertical moving vector of ball
    :return: normalised factor in the same order
    """
    return np.asarray(
        [
            _y_our_paddle / SCREEN_SIZE[1],
            _x_ball / SCREEN_SIZE[0],
            _y_ball / SCREEN_SIZE[1],
            _x_ball_direction,
            _y_ball_direction,
        ]
    )


def perform():
    """
    The main training loop of agent
    :return: graph for performance history
    """
    frame = 0
    history = []

    env = pong_env.PongGame()
    env.init_render()

    _agent = agent.Agent(STATE_COUNT, ACTION_COUNT)

    best_action = 0

    # a random initial state
    state = normalise_state(200.0, 200.0, 200.0, 1.0, 1.0)

    for _frame in range(MAXIMUM_FRAME_COUNT):
        if frame % 100 == 0:
            env.re_render_display(frame, _agent.epsilon)

        best_action = _agent.select_action(state)

        [
            _score,
            _y_our_paddle,
            _x_ball,
            _y_ball,
            _x_ball_direction,
            _y_ball_direction,
        ] = env.take_action(best_action)
        next_state = normalise_state(
            _y_our_paddle, _x_ball, _y_ball, _x_ball_direction, _y_ball_direction
        )

        _agent.record_experience((state, best_action, _score, next_state))
        _agent.train()

        state = next_state

        frame = frame + 1

        if frame % 200 == 0:
            print(
                f"\nFrame: {frame}"
                f"\nScore: {env.score_display: .2f}"
                f"\nEpsilon: {_agent.epsilon}"
            )
            history.append((frame, env.score_display, _agent.epsilon))

    x_val = [item[0] for item in history]
    score_history = [item[1] for item in history]
    epsilon_history = [item[2] for item in history]

    history_dict = {
        'frame_idx': x_val,
        'score': score_history,
        'epsilon': epsilon_history
    }

    plot_training_pong(pd.DataFrame(history_dict))

if __name__ == "__main__":
    perform()
