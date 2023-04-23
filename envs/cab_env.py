"""
    This module contains Cab Environment, which is wrapped to use with OpenAI gym or Gymnasium
    To use the environment out-from-the-box, use:
    `
    from envs.cab_env import CabEnv
    env = CabEnv()
    `

    Or we can register the environment with `gym` or `gymnasium` to use its signature `make` function
    `
    import gym
    from cab_env import CabEnv
    gym.envs.register( id='Cab-v0', entry_point='transp.env.cab_env:CabEnv')
    env = gym.make('Cab-v0')
    `
"""

import sys
import io
import contextlib
import numpy as np
from gym import utils

from gym.envs.toy_text.discrete import DiscreteEnv


class CabEnv(DiscreteEnv):
    """

    **The problem**

    At the beginning of each episode, the passenger spawns randomly at one of these four locations.
    The agent (cab) is responsible for navigating to reach the passenger's location, picking them up,
    and then delivering them to one of the selected destinations.
    The agent must also avoid any walls that are present on the layout.

    ---

    **Usage and modifications**

    *Layout*: The environment can be easily modified by editing the layout in a pre-defined template,
    which currently includes four possible locations for the passenger and vertical walls.
    To ensure that the layout is compatible with environment settings, the top and bottom row (barriers) length
    should be modified to match the number of columns in the grid.

    *Walls*: The template can be edited to add or remove walls or to move them to a different location.
    However, horizontal walls are currently not integrated into the environment
    and will need to be added in the future if required.

    *Rendering colours*: Additionally, we can modify the colour of each location in the playground,
    including the passenger location by editing the RGB values in the `indicator_colours` dictionary.

    *Locations*: To add more locations to the environment, we can simply edit the layout in the template and
    add the new location name to the `location_names` list.

    ---

    **Acknowledgement**

    *The idea*

    The development of this environment took inspiration from the OpenAI Taxi environment, we stimulated the mechanism
    of Taxi Environment. However, reconstructed the code with new coding style and add more comments
    to make it easier to understand and modify.

    This custom environment was created purely for educational purposes, to provide a simple yet effective reinforcement
    learning environment for learners to experiment with and build upon.

    We acknowledge the contributions of the OpenAI team in creating the original Taxi environment,
    and we hope that this custom implementation will advance the field of reinforcement learning education further.

    ---

    *Version compatibility*

    Our custom environment is based on the Discrete environment in OpenAI Gym library 0.21.0.

    There has been a recent update to the original Taxi environment, which is managed by Gymnasium and built on the
    `Env` abstract class from Gym version 0.28.0.
    This new version should be taken into account when we update the environment further.

    For more information about Gymnasium Taxi: <https://www.gymlibrary.dev/environments/toy_text/taxi/>

    ---

    **Further works**

    The authors of this environment intend to update the reward mechanism of the game.

    In general, the design of the reward accumulation function is closely tied to the agent algorithm being used.
    Therefore, setting up the reward accumulation function when designing the agent algorithm is typical.

    However, the author seek to make reward accumulation embedded inside the environment.
    By doing so, the environment can better accommodate more passengers and their desired locations,
    this make the learning algorithm more simple and straightforward.

    """

    # layout of grid
    LAYOUT = [
        "+---------------+",
        "|R: | : : |G: : |",
        "| : : | : | : : |",
        "| : : : : : : : |",
        "| | : : : | : : |",
        "|Y| : | : | : |B|",
        "| : : | : : : : |",
        "| : : | : | : : |",
        "| : : |K: | : : |",
        "+---------------+",
    ]
    location_names = ["R", "G", "Y", "B", "K"]

    # initialise the layout array from string-based
    layout = np.asarray(LAYOUT, dtype="c")

    """
    The status indicator colors:
    'empty_cab': the location of the cab when there is no passenger inside
    'loaded_cab': after the cab load passenger, its colour will change
    'passenger_out': this colour mark the location of passenger at selected location
    'destination': indicate the desired destination which the cab should take the passenger to
    """

    indicator_colours = {
        "empty_cab": "yellow",
        "loaded_cab": "green",
        "passenger_out": "blue",
        "destination": "magenta",
    }

    # define the actions
    action_dict = {
        0: "South",
        1: "North",
        2: "East",
        3: "West",
        4: "Pick-up",
        5: "Drop-off",
    }
    actions = list(action_dict.keys())
    action_labels = list(action_dict.values())

    # reward/ penalty dictionary
    reward_dict = {"step": -1, "penalty": -30, "final_reward": 60}

    def __init__(self):
        # scan the layout and define location coordinates
        self.locations = []
        for location_name in self.location_names:
            [[y_temp, x_temp]] = np.argwhere(
                self.layout == bytes(location_name, encoding="utf-8")
            )
            y_location, x_location = int(y_temp - 1), int((x_temp - 1) / 2)
            self.locations.append((y_location, x_location))
        self.num_location = len(self.locations)
        self.location_ids = list(range(self.num_location))

        # initialise attributes as size of the layout
        self.num_x = int((len(self.layout[1, :]) - 1) / 2)
        self.num_y = int(len(self.layout[:, 1]) - 2)
        self.x_max = self.num_x - 1
        self.y_max = self.num_y - 1

        # passenger state spaces
        # state of passenger: in the locations and in the cab
        self.passenger_ids = list(range(len(self.locations) + 1))

        # state spaces
        self.state_count = (
            self.num_x * self.num_y * len(self.locations) * len(self.passenger_ids)
        )
        self.init_state_distribution = np.zeros(self.state_count)

        # init state-action map
        self.P = {
            state: {action: [] for action in self.actions}
            for state in range(self.state_count)
        }

        # init environment mechanism
        for _y in range(self.num_y):
            for _x in range(self.num_x):
                for passenger_id in self.passenger_ids:
                    for destination_id in self.location_ids:
                        state = self.generate_state_id(
                            _y, _x, passenger_id, destination_id
                        )

                        # passenger is at pick-up location
                        if (
                            passenger_id < self.num_location
                            and passenger_id != destination_id
                        ):
                            self.init_state_distribution[state] += 1

                        for action in self.actions:
                            # default: not moving, no pick-up/dr op-off, no termination
                            new_y, new_x, new_passenger_id = _y, _x, passenger_id
                            reward = self.reward_dict.get("step")
                            termination = False
                            cab_location = (_y, _x)

                            # action conditions
                            if action == 0:
                                new_y = min(_y + 1, self.x_max)

                            elif action == 1:
                                new_y = max(_y - 1, 0)

                            elif (
                                action == 2 and self.layout[_y + 1, _x * 2 + 2] == b":"
                            ):
                                new_x = min(_x + 1, self.y_max)

                            elif action == 3 and self.layout[_y + 1, _x * 2] == b":":
                                new_x = max(_x - 1, 0)

                            # pick-up
                            elif action == 4:
                                # pick-up correctly (the last passenger_id is when passenger in the cab)
                                if (passenger_id < self.num_location) and (
                                    cab_location == self.locations[passenger_id]
                                ):
                                    new_passenger_id = self.passenger_ids[-1]
                                # pick-up at wrong location
                                else:
                                    reward = self.reward_dict.get("penalty")

                            # drop-off
                            elif action == 5:
                                # drop-off correctly
                                if (
                                    cab_location == self.locations[destination_id]
                                ) and (passenger_id == self.passenger_ids[-1]):
                                    new_passenger_id = destination_id
                                    termination = True
                                    reward = self.reward_dict.get("final_reward")
                                # drop off in selected location but not in destination
                                elif (cab_location in self.locations) and (
                                    passenger_id == self.passenger_ids[-1]
                                ):
                                    new_passenger_id = self.locations.index(
                                        cab_location
                                    )
                                # drop off outside selected location
                                else:
                                    reward = self.reward_dict.get("penalty")

                            # assign the state
                            new_state = self.generate_state_id(
                                new_y, new_x, new_passenger_id, destination_id
                            )

                            # fill in state-action map
                            # as the environment is deterministic, we set the probability 1.0
                            self.P[state][action].append(
                                (1.0, new_state, reward, termination)
                            )

        self.init_state_distribution /= self.init_state_distribution.sum()
        DiscreteEnv.__init__(
            self,
            self.state_count,
            len(self.actions),
            self.P,
            self.init_state_distribution,
        )

    def generate_state_id(self, _y, _x, passenger_id, destination_id):
        """
        This function generates a state index using the cab's coordinates, passenger status, and destination id
        :param _y: current row (y-coordinate) of the cab
        :param _x: current col (x-coordinate) of the cab
        :param passenger_id: status of passenger (in locations and in the cab)
        :param destination_id: id of desired destination
        :return:
        """
        _id = _y
        _id *= self.num_x
        _id += _x
        _id *= len(self.passenger_ids)
        _id += passenger_id
        _id *= self.num_location
        _id += destination_id
        return _id

    def get_state_from_id(self, state_id):
        """
        This function to return the cab's coordinates, passenger status, and destination id from state index
        :param state_id: index of the current state
        :return: (y, x, passenger_id, destination_id)
        """
        state = [(state_id % len(self.locations))]
        state_id //= len(self.locations)
        state.append(state_id % len(self.passenger_ids))
        state_id //= len(self.passenger_ids)
        state.append(state_id % self.num_x)
        state_id //= self.num_x
        state.append(state_id)
        assert 0 <= state_id < self.num_y
        return list(reversed(state))

    def render(self, mode="human"):
        """
        Renders the current state of the environment for easier understanding by visualisation
        Note that we override default render function of `DiscreteEnv`, eliminating `mode` (or `render_mode`)
        since we only have 1 way of rendering
        :return: layout of current state as text
        """
        # initialise output, we only print to console
        if mode == "ansi":
            display = io.StringIO()
        else:
            display = sys.stdout

        # obtain the display layout for rendering
        d_layout = self.layout.copy().tolist()
        d_layout = [[e.decode("utf-8") for e in line] for line in d_layout]

        # get state from id
        _y, _x, passenger_id, destination_id = self.get_state_from_id(self.s)

        # if passenger in the cab we will render * for the cab
        icon = lambda e: "*" if e == " " else e

        # if passenger is not in the cab
        if passenger_id < len(self.locations):
            # render the cab color
            d_layout[1 + _y][2 * _x + 1] = utils.colorize(
                d_layout[1 + _y][2 * _x + 1],
                self.indicator_colours.get("empty_cab"),
                highlight=True,
            )

            # render passenger location color
            x_passenger, y_passenger = self.locations[passenger_id]
            d_layout[1 + x_passenger][2 * y_passenger + 1] = utils.colorize(
                d_layout[1 + x_passenger][2 * y_passenger + 1],
                self.indicator_colours.get("passenger_out"),
                bold=True,
            )

        # if passenger is in the cab
        else:
            # render the cab color
            d_layout[1 + _y][2 * _x + 1] = utils.colorize(
                icon(d_layout[1 + _y][2 * _x + 1]),
                self.indicator_colours.get("loaded_cab"),
                highlight=True,
            )

        # render destination color
        x_destination, y_destination = self.locations[destination_id]
        d_layout[1 + x_destination][2 * y_destination + 1] = utils.colorize(
            d_layout[1 + x_destination][2 * y_destination + 1],
            self.indicator_colours.get("destination"),
            bold=False,
        )

        # print to file
        display.write("\n".join(["".join(row) for row in d_layout]) + "\n")
        if self.lastaction is not None:
            display.write(f"Action: {self.action_labels[self.lastaction]}")
        else:
            display.write("\n")

        # if render mode is ansi, we need to close the file
        if mode == "ansi":
            with contextlib.closing(display):
                return display.getvalue()
