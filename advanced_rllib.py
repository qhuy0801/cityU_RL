"""
Module for running Ray RLlib on the Atari game 'BreakoutNoFrameskip-v4'.
Example usage:
    Run main method
    Run `%load_ext tensorboard` in ipython command prompt
    Run `%tensorboard --logdir logs/<ENV_NAME>` in ipython command prompt
    Observe the training progress using browser
"""
import ray
from ray.tune import run

ENV_NAME = "BreakoutNoFrameskip-v4"


def run_rllib():
    """
    Configuration and run Ray RLlib on selected environment
    :return:
    """
    config = {
        "env": ENV_NAME,
        "run": "DQN",
        "stop": {"episode_reward_mean": 10, "time_total_s": 7200},
        "config": {
            "framework": "torch",
            "frameskip": 1,
            "env_config": {"nondeterministic": "false", "frameskip": 1},
            "double_q": "true",
            "dueling": "true",
            "noisy": "false",
            "prioritized_replay": "false",
            "num_atoms": 1,
            "gamma": 0.99,
            "lr": 0.0000625,
            "learning_starts": 20000,
            "buffer_size": 1000000,
            "adam_epsilon": 0.00015,
            "hiddens": [512],
            "target_network_update_freq": 8000,
            "epsilon_timesteps": 200000,
            "sample_batch_size": 4,
            "train_batch_size": 32,
            "schedule_max_timesteps": 2000000,
            "exploration_final_eps": 0.01,
            "exploration_fraction": .1,
            "prioritized_replay_alpha": 0.5,
            "beta_annealing_fraction": 1.0,
            "final_prioritized_replay_beta": 1.0,
            "num_gpus": 0.2,
            "timesteps_per_iteration": 10000
        },
    }

    stop = {"episode_reward_mean": 5}

    ray.shutdown()
    ray.init(
        num_cpus=3,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )
    # execute training
    run(
        "DQN",
        config=config,
        stop=stop,
        checkpoint_at_end=True,
        local_dir="BreakoutNoFrameskip-v4",
    )

if __name__ == "__main__":
    run_rllib()
