# we define a minimum epsilon
epsilon_end = 0.001

def get_epsilon(
    epsilon_start, num_episode, current_episode, strategy='exponential', epsilon_decay=None
):
    if strategy == 'linear':
        epsilon_step = epsilon_start / num_episode
        epsilon_ = epsilon_start - epsilon_step * (current_episode - 1)
        epsilon = max(epsilon_, epsilon_end)
    elif strategy == 'exponential':
        if epsilon_decay is None:
            raise ValueError('Decay rate required for exponential strategy')
        epsilon_ = epsilon_start * (epsilon_decay**current_episode)
        epsilon = max(epsilon_, epsilon_end)
    else:
        raise ValueError('The strategy should be linear or exponential')
    return epsilon
