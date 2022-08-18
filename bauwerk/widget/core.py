"""Main widget module."""

import sys
import gym
import threading
import time
import numpy as np
import traitlets
import ipywidgets as widgets
import matplotlib.pyplot as plt
from loguru import logger

import bauwerk
from bauwerk.constants import PROJECT_PATH

bauwerk.setup()


class Game(widgets.VBox):
    """Bauwerk building control game widget."""

    def __init__(
        self,
        log_level: str = "error",
        height: str = "400px",
        step_time=None,
        visible_steps=24,
    ):
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

        self.visible_steps = visible_steps

        self.active_thread = None
        self.stop_requested = False

        # Setting up controller
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
        self.menu_buttons = self._setup_menu_buttons()

        # Setting up main figure

        # This sets the first observations
        self.reset()

        self._setup_figure()

        self.out = widgets.Output(
            layout={
                "width": "100px",
                "height": "100px",
            }
        )
        self.hidden_out = widgets.Output()

        with self.out:
            print("Get ready!")

        # children = [self.control, self.vis, self.out]
        self.main_app = widgets.AppLayout(
            left_sidebar=self.control,
            center=self.fig.canvas,
            footer=self.out,
            pane_widths=[1, 9, 0],
        )
        super().__init__(children=[self.menu_buttons, self.main_app])

        self.game_finished = False

        # Setup automatic stepping
        self.step_time = step_time
        if self.step_time:
            self.add_traits(step_requested=traitlets.Bool().tag(sync=True))
            self.step_requested = False
            self.observe(
                self._process_step_request, names="step_requested", type="change"
            )
            # self._launch_update_requesting_thread()
        else:
            self.control.observe(self.step, names="value")

    def _setup_menu_buttons(self):
        # Setting up menu
        self.start_button = widgets.Button(
            description="Start",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Start game",
            icon="play",  # (FontAwesome names without the `fa-` prefix)
        )
        self.start_button.on_click(self._process_start_request)

        self.stop_button = widgets.Button(
            description="Stop",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Stop game",
            icon="stop",  # (FontAwesome names without the `fa-` prefix)
        )
        self.stop_button.on_click(self._process_stop_request)

        self.reset_button = widgets.Button(
            description="Reset",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Reset game.",
            icon="restart",  # (FontAwesome names without the `fa-` prefix)
        )
        self.reset_button.on_click(self._process_reset_request)

        return widgets.HBox(
            children=[self.start_button, self.stop_button, self.reset_button]
        )

    def reset(self):

        obs = self.env.reset()
        self.obs_values = {
            key: [np.array([0], dtype=np.float32)] * self.visible_steps
            for key in obs.keys()
            if key != "time_step"
        }
        self.add_obs(obs)

    def add_obs(self, obs):
        for key in self.obs_values.keys():
            self.obs_values[key].append(obs[key])

    def _setup_figure(self):

        # Setting up figure
        with plt.ioff():
            with plt.xkcd(scale=1, length=20000, randomness=2):
                # Setting correct height in pixels
                # Conversion following setup described in:
                # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
                px = 1 / plt.rcParams["figure.dpi"]
                fig_height = self.height_px * px * 1  # in inches

                self.fig = plt.figure(constrained_layout=True, figsize=(7, fig_height))
                self.fig.canvas.header_visible = False
                self.fig.canvas.toolbar_visible = False
                self.fig.canvas.resizable = False
                self.fig.canvas.footer_visible = False
                # self.fig.canvas.layout.height = "200px"
                # self.fig.canvas.layout.width = "400px"

                subfigs = self.fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 2])
                ax_left = subfigs[0].subplots(1)
                ax_left.axis("off")

                self.img_house = plt.imread(PROJECT_PATH / "widget/house.png")
                ax_left.imshow(self.img_house)
                self.obs_axs = subfigs[1].subplots(len(self.obs_values))

                self.obs_lines = []
                self.line_x = np.linspace(0, self.visible_steps, self.visible_steps)

                for i, (obs_name, obs_part) in enumerate(self.obs_values.items()):
                    self.obs_lines.append(
                        self.obs_axs[i].plot(
                            self.line_x,
                            obs_part[-self.visible_steps :],
                        )
                    )
                    self.obs_axs[i].set_title(obs_name.replace("_", " "))
                for ax in self.obs_axs:
                    ax.label_outer()

    def _update_figure(self):
        for i, obs_part in enumerate(self.obs_values.values()):
            # setting new data
            self.obs_lines[i][0].set_data(self.line_x, obs_part[-self.visible_steps :])

            # rescaling y axis
            # based on https://stackoverflow.com/a/7198623
            axs = self.obs_axs[i]
            axs.relim()
            axs.autoscale_view(True, True, True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _launch_update_requesting_thread(self, change=None):
        # pylint: disable=unused-argument
        def work(widget):
            max_steps = 25
            for _ in range(max_steps):
                if widget.game_finished or widget.stop_requested:
                    break
                time.sleep(self.step_time)
                widget.step_requested = True

        thread = threading.Thread(target=work, args=(self,))
        thread.start()
        logger.info("Started thread.")
        return thread

    def _process_start_request(self, change=None):
        # pylint: disable=unused-argument
        if self.active_thread:
            self.stop_requested = True
            self.active_thread.join()
            self.stop_requested = False

        self.active_thread = self._launch_update_requesting_thread()

    def _process_stop_request(self, change=None):
        # pylint: disable=unused-argument
        if self.active_thread:
            self.stop_requested = True
            self.active_thread.join()
            self.stop_requested = False

    def _process_reset_request(self, change=None):
        # pylint: disable=unused-argument
        self._process_stop_request()
        self.reset()
        self._update_figure()

    def _process_step_request(self, change):
        # pylint: disable=unused-argument
        if self.step_requested:
            logger.info("Step requested.")
            self.step()
            self.step_requested = False
        else:
            logger.info("widget.step_requested is false.")
            pass

    def step(self, change=None):
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

            self._update_figure()
