"""Utilities for plotting experimental results."""

import copy
import seaborn as sns

sns.set_theme(style="white", context="paper", font="serif")
palette = sns.color_palette("deep")


def get_rel_perf(maximum, minimum, perf):
    return (perf - minimum) / (maximum - minimum)


def get_loc(house, idx, height, num_values_per_house, space_between_graphs):
    """Get location of bar in plot for perf measure 'idx' in building 'house'."""
    return house - height * (num_values_per_house / 2 - 0.5) + height * idx


def create_bar_chart(
    env_data,
    max_key=None,
    min_key=None,
    remove_keys=None,
    include_legend=True,
    file_name="test.png",
    ax=None,
    absolute=False,
    title=None,
    x_label=None,
    space_between_graphs=0.1,
):
    if ax is None:
        # Figure Size
        fig, ax = plt.subplots(figsize=(4.5, 5.5))

    ys = []
    y_labels = []
    nocharge_lines = []

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
        space_between_houses = 2
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

    try:
        return fig
    except:
        return None
