"""Main widget module."""

import sys
import gym
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from loguru import logger

import bauwerk
from bauwerk.constants import PROJECT_PATH

bauwerk.setup()


class Game(widgets.HBox):
    """Bauwerk building control game widget."""

    def __init__(self, log_level: str = "error", height: str = "400px"):
        """Bauwerk building control game widget.

        Args:
            log_level (str, optional): Lower case logging level (e.g. "info", "debug").
                Defaults to "error".
            height (str, optional): height of widget in px as str ending with px.
                Defaults to "400px".
        """

        # Setup logging
        logger.remove()
        logger.add(sys.stderr, level=log_level.upper())
        plt.set_loglevel(log_level)

        # Create underlying env
        self.env = gym.make(
            "bauwerk/SolarBatteryHouse-v0",
            new_step_api=True,
        )

        self.height = height
        self.height_px = int(height.replace("px", ""))
        action_high = self.env.action_space.high[0]
        action_low = self.env.action_space.low[0]

        self.control = widgets.FloatSlider(
            description="Battery",
            orientation="vertical",
            min=action_low,
            max=action_high,
            continuous_update=False,
            layout={"height": height},
        )
        self.control.observe(self.step, names="value")
        self.vis = widgets.Output(layout={"width": "600px", "height": height})
        self.out = widgets.Output(
            layout={
                "width": "100px",
                "height": "100px",
            }
        )
        self.hidden_out = widgets.Output()

        with self.out:
            print("Get ready!")

        children = [self.control, self.vis, self.out]
        super().__init__(children=children)

        self.img_house = plt.imread(PROJECT_PATH / "widget/house.png")

        self.game_finished = False
        self.reset()

    def reset(self):

        obs = self.env.reset()
        self.obs_values = {
            key: [np.array([0], dtype=np.float32)]
            for key in obs.keys()
            if key != "time_step"
        }
        self.add_obs(obs)
        self.plot()

    def add_obs(self, obs):
        for key in self.obs_values.keys():
            self.obs_values[key].append(obs[key])

    def plot(self):

        with plt.xkcd(scale=1, length=20000, randomness=2):
            self.vis.clear_output(wait=True)
            with self.vis:

                # Setting correct height in pixels
                # Conversion following setup described in:
                # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
                px = 1 / plt.rcParams["figure.dpi"]
                fig_height = self.height_px * px  # in inches

                fig = plt.figure(constrained_layout=True, figsize=(8, fig_height))
                subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 2])
                ax_left = subfigs[0].subplots(1)
                ax_left.axis("off")
                ax_left.imshow(self.img_house)
                axs = subfigs[1].subplots(len(self.obs_values))
                for i, (obs_name, obs_part) in enumerate(self.obs_values.items()):
                    axs[i].plot(obs_part)
                    axs[i].set_title(obs_name.replace("_", " "))
                for ax in axs:
                    ax.label_outer()
                plt.show()

    def step(self, change):
        # pylint: disable=unused-argument

        if not self.game_finished:

            self.out.clear_output()

            action = self.control.value
            # with self.out:
            # print("Action", action)
            action = np.array([action], dtype=np.float32)
            # pylint: disable=unused-variable
            observation, reward, terminated, truncated, info = self.env.step(action)

            self.add_obs(observation)

            with self.out:
                # print(observation)

                if terminated or truncated:
                    self.game_finished = True
                    self.control.set_trait("disabled", True)
                    print("Game over.")

            self.plot()
