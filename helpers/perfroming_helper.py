import numpy as np


def cab_perform(env, q_table, reward_dict, num_episodes):
    sequences = []
    for _ in range(num_episodes):
        state = env.reset()
        total_step, penalties, total_reward = 0, 0, 0

        termination = False

        while not termination:
            action = np.argmax(q_table[state])
            state, reward, termination, _ = env.step(action)

            if reward == int(reward_dict.get("penalty")):
                penalties += 1

            total_reward += reward

            # Put each rendered frame into dict for animation
            sequences.append(
                {
                    "_rendered": env.render(mode="ansi"),
                    "_state": state,
                    "_action": action,
                    "_total_reward": total_reward,
                }
            )
            total_step += 1

    return sequences
