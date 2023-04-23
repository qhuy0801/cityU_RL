import numpy as np

from envs.cab_env import CabEnv


class CabEnvV2(CabEnv):
    # layout of grid
    LAYOUT = [
        "+-----------------+",
        "|R: | : : |G: | : |",
        "| : : | : | : : : |",
        "| : : : : : : : : |",
        "| | : : : | : : : |",
        "|Y| : | : | : |B: |",
        "| : : | | : : : : |",
        "| : : | : | : : : |",
        "| : : |K: | : : | |",
        "| : : | : | : : |M|",
        "+-----------------+",
    ]
    layout = np.asarray(LAYOUT, dtype="c")
    location_names = ["R", "G", "Y", "B", "K", "M"]

    def __init__(self):
        super().__init__()
