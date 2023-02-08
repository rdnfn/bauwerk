"""Utilities for plotting experimental results."""

from __future__ import annotations
import copy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set_theme(style="white", context="paper", font="serif")
palette = sns.color_palette("deep")


def get_rel_perf(maximum, minimum, perf):
    return (perf - minimum) / (maximum - minimum)


def get_loc(house, idx, height, num_values_per_house, space_between_graphs):
    """Get location of bar in plot for perf measure 'idx' in building 'house'."""
    return house - height * (num_values_per_house / 2 - 0.5) + height * idx


def create_bar_chart(
    env_data: dict,
    max_key: str = None,
    min_key: str = None,
    remove_keys: list = None,
    include_legend: bool = True,
    ax: object = None,
    absolute: bool = False,
    title: str = None,
    x_label: str = None,
    space_between_houses: float = 1.0,
    space_between_graphs: float = 0.1,
) -> object:
    """Plot bar chart of experimental results.

    Args:
        env_data (dict): results as dictionary with structure
            dict[house_key][alg_name].
        max_key (str, optional): if not using absolute data,
            key of performance normalised to 1. Defaults to None.
        min_key (str, optional):if not using absolute data,
            key of performance normalised to 0. Defaults to None.
        remove_keys (list, optional): keys of algorithms to
            be removed from env_data. Defaults to None.
        include_legend (bool, optional): whether to include
            legend in figure. Defaults to True.
        ax (object, optional): ax to build figure in.
            Defaults to None. If None new figure is created.
        absolute (bool, optional): whether the figure should use
            absolute as opposed to relative values. Defaults to False.
        title (str, optional): title of figure. Defaults to None.
        x_label (str, optional): label of x axis. Defaults to None.
        space_between_houses (float, optional): space between bar chart
            groups (each a house) in individual bar widths.
            Defaults to 1.0.
        space_between_graphs (float, optional): space between graphs.
            Proportional of height of one algorithm in plot.
            Defaults to 0.1.

    Returns:
        object: either returns new matplotlib figure
            or axis (if ax given).
    """
    if ax is None:
        # Figure Size
        ax_given = False
        fig, ax = plt.subplots(figsize=(4.5, 5.5))
    else:
        ax_given = True

    ys = []
    y_labels = []

    # Create consistent color code for each method
    col_code = {}
    for i, key in enumerate(env_data[1].keys()):
        col_code[key] = i

    for i, size in enumerate(env_data.keys()):
        if isinstance(size, int):
            name = f"{size}kWh"
        else:
            name = size

        perf_dict: dict = copy.deepcopy(env_data[size])

        # Remove algorithms that don't fit
        if max_key is not None:
            perf_dict.pop(max_key)
        if min_key is not None:
            perf_dict.pop(min_key)
        if remove_keys is not None:
            for key in remove_keys:
                perf_dict.pop(key)

        num_values_per_house = len(perf_dict.keys())
        height = 1 / ((num_values_per_house + space_between_houses))

        # Add bars for each algorithm and each house
        for j, (key, value) in enumerate(perf_dict.items()):
            if not absolute:
                rel_value = get_rel_perf(
                    maximum=env_data[size][max_key],
                    minimum=env_data[size][min_key],
                    perf=env_data[size][key],
                )
            else:
                rel_value = env_data[size][key]
            ax.barh(
                get_loc(i, j, height, num_values_per_house, space_between_graphs),
                width=rel_value,
                height=height * (1 - space_between_graphs / 2),
                color=palette[col_code[key]],
                label=key,
            )

        ys.append(i)
        y_labels.append(name)

    # Add annotation to bars
    x_len = ax.get_xbound()[1] - ax.get_xbound()[0]
    for i in ax.patches:
        width = i.get_width()
        ax.text(
            width + x_len * 0.007 * np.sign(width),
            i.get_y() + i.get_height() * 0.55,
            str(round((width), 3)),
            fontsize=7,
            color="black",
            horizontalalignment=("left" if width > 0 else "right"),
            verticalalignment="center",
        )

    # add the battery size labels
    ax.set_yticks(ys, y_labels)
    if not absolute:
        ax.set_xticks(
            [0, 1],
            [f"({min_key}) 0", f"({max_key}) 1"],
            rotation=35,
            horizontalalignment="right",
        )
        ax.tick_params(axis="x", pad=-10)  # reduce the padding of tick labels
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # add axis labels and title
    ax.set_ylabel("Building's battery size")
    if not absolute and title is None:
        title = f"Rel. to {min_key} and {max_key}"
    if x_label is not None:
        ax.set_xlabel(x_label)

    ax.set_title(title)

    if include_legend:
        # avoid duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower left")

    # change order to from smallest to largest battery size
    ax.invert_yaxis()

    # add vertical lines if not absolute
    if not absolute:
        ax.vlines(
            [0, 1],
            *ax.get_ylim(),
            colors=["grey", "grey"],
            linestyles=["solid", "dotted"],
        )

    # remove default frame around figure
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # extend figure slightly to left to show full vline at 0
    ax.set_xlim(left=ax.get_xlim()[0] - 0.005)

    if ax_given:
        return ax
    else:
        return fig
