import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output as clear


def visualise_step_epsilon(training_info, tile):
    # get only first 500 record for clearer visualisation
    data = training_info.head(500)

    sns.set(rc={"figure.figsize": (8, 6)})

    # plot visualisation
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1 = sns.lineplot(
        data=data,
        x="episode",
        y="num_steps",
        color="c",
        label="Number of steps",
        legend=False,
    )
    ax_1.grid(False)
    ax_2 = ax_1.twinx()
    ax_2 = sns.lineplot(
        data=data, x="episode", y="epsilon", color="r", label="Epsilon", legend=False
    )
    ax_2.grid(False)
    fig.legend()
    fig.suptitle(tile, fontsize=14)
    plt.show()


def multiple_line_plot(training_info, title, x_label, y_label, legend_title=None):
    sns.lineplot(
        x="episode",
        y="value",
        hue="variable",
        data=pd.melt(training_info, ["episode"]),
        alpha=0.6,
        palette="tab10",
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend_title if legend_title is not None else "variable")
    plt.show()


def display_(sequences) -> None:
    for _, sequence in enumerate(sequences):
        clear(wait=True)
        print(sequence["_rendered"])
        print(f"State: {sequence['_state']}")
        print(f"Action: {sequence['_action']}")
        print(f"Reward: {sequence['_total_reward']}")
        time.sleep(1)

def display_atari(img):
    plt.imshow(img.astype(int))
    plt.show()

def plot_training_pong(data):
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Score')
    ax1.plot(data['score'], color='C1', label='score')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Epsilon')
    ax2.plot(data['epsilon'], color='C3', label='epsilon')
    fig.legend()
    plt.show()
