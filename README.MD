# Reinforcement learning
## City, University of London
Email: [quang.bui.2@city.ac.uk](mailto:quang.bui.2@city.ac.uk)

# Introduction
This repository contains different sub-projects that implement various reinforcement learning algorithms:

1. **Q-Learning:** A basic algorithm that uses an action-value Q-table. The algorithm is implemented in a custom environment based on the OpenAI Gym Taxi environment as we called Cab.
2. **Deep Q-Network (DQN):** An implementation of the DQN algorithm on a custom Pong environment. The environment uses Pygame to render the game and Keras for the neural network architecture.
3. **RLlib DQN:** An implementation of the DQN algorithm using Ray's RLlib library. The implementation includes Double Q-learning and a noise layer for exploration.

# Project structure
We've got three sub-projects in the repo, labelled 1, 2, and 3 (as we mentioned earlier). Each has its own folder with everything you need to run the code and make the results happen.
```bash
├── main.py
├── algo
│   ├── basic_q_learning (1)
│   │   └── q_learning.py
│   │
│   ├── dqn_pygame_pong (2)
│   │   ├── agent.py
│   │   ├── dqn.py
│   │   └── replay_memory.py
│   │
│   └── e_greedy (1)
│       └── epsilon_greedy.py
│
├── envs
│   ├── cab_env.py (1)
│   ├── cab_env_v2 (*)
│   └── pong_env.py
│
├── helpers
│   ├── performing_helper.py
│   └── visualising_helper.py
│
├── basic_task_program (1)
│
├── dqn_pong_perform.py (2)
│
├── advanced_rllib.py (3)
│
├── README.md
│
└── requirements.txt

(*) `cab_env_v2` is just an implemetation that we tested our coding style
(*) it is a more complex environment than `cab_env` and traditional Taxi environment
(*) we can use it as a template to further modify this environment
(*) however, `cab_env_v2` is not used in this research

```


```
pip install -r /path/to/requirements.txt
```

```
pip install "gym[atari]"
pip install "gym[accept-rom-license]"
pip install gymnasium
pip install shimmy
```

```
%load_ext tensorboard
%tensorboard --logdir logs/path
```

### License
This project does not currently have a registered license. Please note that this is an individual project for educational purposes only. We have referenced some license-free code for algorithmic ideas, but all code in this repository has been completely rewritten.
We referenced some source:

[Jordi Torres Deep - Reinforcement Learning Explained](https://torres.ai/deep-reinforcement-learning-explained-series/)

[JulesVerny/PongReinforcementLearning](https://github.com/JulesVerny/PongReinforcementLearning)
