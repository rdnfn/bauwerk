"""Main widget module."""

import sys
import gym
import threading
import time
import numpy as np
import traitlets
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger

import bauwerk
from bauwerk.constants import PROJECT_PATH

bauwerk.setup()


class Game(widgets.VBox):
    """Bauwerk building control game widget."""

    def __init__(
        self,
        log_level: str = "error",
        height: int = 500,
        step_time=None,
        visible_steps=24,
        episode_len=168,
    ):
        """Bauwerk building control game widget.

        Args:
            log_level (str, optional): Lower case logging level (e.g. "info", "debug").
                Defaults to "error".
            height (int, optional): height of widget in px.
                Defaults to 400 px.
        """

        # Setup logging
        logger.remove()
        logger.add(sys.stderr, level=log_level.upper())
        plt.set_loglevel(log_level)

        # Create underlying env
        self.env = gym.make(
            "bauwerk/SolarBatteryHouse-v0", new_step_api=True, episode_len=episode_len
        )

        logo_file = open(PROJECT_PATH / "widget/bauwerk_game_logo.png", "rb")
        self.game_logo_img = logo_file.read()

        self.episode_len = episode_len

        self.fig_height = height - 150

        self.visible_steps = visible_steps

        self.reward_label = "reward (payment)"

        self.active_thread = None
        self.pause_requested = False

        # Setting up controller
        action_high = self.env.action_space.high[0]
        action_low = self.env.action_space.low[0]

        self.control = widgets.FloatSlider(
            description="Battery",
            orientation="vertical",
            min=action_low,
            max=action_high,
            step=0.05,
            continuous_update=True,
            layout={"height": f"{self.fig_height}px"},
        )
        self.menu_buttons = self._setup_menu_buttons()

        # Setting up main figure

        # This sets the first observations
        self.reset()

        self._setup_figure()

        self.game_lower_part = widgets.AppLayout(
            left_sidebar=self.control,
            center=self.fig.canvas,
            pane_widths=[1, 9, 0],
        )
        self.heading = widgets.HTML(
            value=(
                # "<code style='color: black'><h1 style='display: inline'>Bauwerk Game"
                # "</h1>&nbsp;&nbsp;&nbsp;"
                "<h3 style='display: inline'>Level: "
                "SolarBatteryHouse-v0</h3></code>"
            ),
        )
        self.main_app = widgets.VBox(
            [
                widgets.VBox(
                    [
                        self.menu_buttons,
                        # self.heading,
                    ]
                ),
                self.game_lower_part,
            ]
        )
        self.main_app.layout.display = "none"

        self._setup_start_screen()

        super().__init__(
            children=[self.start_screen, self.main_app],
            layout={
                "height": f"{height}px",
                "align_items": "center",
            },
        )

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

        self.pause_button = widgets.Button(
            description="Pause",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Pause game",
            icon="pause",  # (FontAwesome names without the `fa-` prefix)
        )
        self.pause_button.on_click(self._process_pause_request)

        self.reset_button = widgets.Button(
            description="Reset",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Reset game.",
            icon="refresh",  # (FontAwesome names without the `fa-` prefix)
        )
        self.reset_button.on_click(self._process_reset_request)

        self.back_to_menu_button = widgets.Button(
            description="Menu",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Back to main menu.",
            icon="bars",  # (FontAwesome names without the `fa-` prefix)
        )
        self.back_to_menu_button.on_click(self._go_to_start_screen)

        self.game_logo_small = widgets.Image(
            value=self.game_logo_img,
            format="png",
            width=50,
            layout={"margin": "0px 10px 0px 0px"},
        )

        return widgets.HBox(
            children=[
                self.game_logo_small,
                self.start_button,
                self.pause_button,
                self.reset_button,
                self.back_to_menu_button,
            ],
            # layout={"align_items": "center"},
        )

    def _setup_figure(self):

        # Setting up figure
        with plt.ioff():
            # with plt.xkcd(scale=1, length=20000, randomness=2):

            # Setting correct height in pixels
            # Conversion following setup described in:
            # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
            px = 1 / plt.rcParams["figure.dpi"]
            fig_height = self.fig_height * px * 1  # in inches

            self.fig = plt.figure(
                constrained_layout=True,
                figsize=(7, fig_height),  # dpi=50
            )
            self.fig.canvas.header_visible = False
            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.resizable = False
            # self.fig.canvas.footer_visible = False
            # self.fig.canvas.layout.height = "200px"
            # self.fig.canvas.layout.width = "400px"

            plt.rcParams.update({"font.size": 10})

            subfigs = self.fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 2])

            # Left handside of plt animation
            axs_left = subfigs[0].subplots(2, gridspec_kw={"height_ratios": [2, 1]})
            self._create_house_figure(axs_left[0])

            # Draw text
            axs_left[1].axis("off")
            self.score_text = axs_left[1].text(
                x=0.1,
                y=0.7,
                s="Score: 0",
                # animated=True,
                fontfamily="monospace",
                fontsize=16,
            )

            # Right handside of plt animation
            # Create observation data plots
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

    def _create_house_figure(self, img_ax):
        img_ax.axis("off")
        self.img_house = plt.imread(PROJECT_PATH / "widget/house_v2.png")
        self.indicator_solar = img_ax.add_patch(
            mpatches.Circle(
                (111, 72),
                radius=75,
                alpha=0.0,
                facecolor="white",
            )
        )
        # Parallelogram of sun rays
        x = [133, 273, 304, 208, 110]
        y = [274, 274, 198, 120, 166]
        self.indicator_solar_ray_xy = np.array(list(zip(x, y)))
        self.indicator_solar_ray = mpatches.Polygon(
            xy=self.indicator_solar_ray_xy,
            facecolor="white",
            alpha=0.3,
        )
        img_ax.add_patch(self.indicator_solar_ray)

        self.indicator_load = img_ax.add_patch(
            mpatches.Rectangle(
                (143, 347), width=99, height=92, facecolor="black", alpha=0.5
            )
        )
        self.indicator_battery = img_ax.add_patch(
            mpatches.Rectangle(
                (344, 331), width=29, height=80, facecolor="white", alpha=0.7
            )
        )
        # Parallelogram of battery
        x = [384, 421, 421, 384]
        y = [328, 300, 380, 408]
        self.indicator_battery_side_xy = np.array(list(zip(x, y)))
        self.indicator_battery_side = mpatches.Polygon(
            xy=self.indicator_battery_side_xy,
            facecolor="white",
            alpha=0.7,
        )
        img_ax.add_patch(self.indicator_battery_side)

        img_ax.imshow(self.img_house)

    def _update_house_figure(self):
        def get_alpha_value(obs_val, data_source, min_alpha=0.1, max_alpha=0.9):
            val_range = data_source.max_value - data_source.min_value
            alpha = float((obs_val - data_source.min_value) / val_range)
            alpha = min_alpha + alpha * (max_alpha - min_alpha)
            return alpha

        # updating figure
        solar_strength = get_alpha_value(
            self.obs_values["pv_gen"][-1], self.env.solar, 0.2, 0.9
        )
        solar_ray_strength = get_alpha_value(
            self.obs_values["pv_gen"][-1], self.env.solar, 0.0, 0.5
        )
        self.indicator_solar.set_alpha(1 - solar_strength)
        self.indicator_solar_ray.set_alpha(solar_ray_strength)

        load_strength = get_alpha_value(
            self.obs_values["load"][-1], self.env.load, 0.4, 1
        )

        self.indicator_load.set_alpha(1 - load_strength)

        battery_cont = float(
            self.obs_values["battery_cont"][-1] / self.env.battery.size
        )

        ba_height = 80 * battery_cont
        ba_y = 330 + 80 - ba_height

        self.indicator_battery.set_height(ba_height)
        self.indicator_battery.set_y(ba_y)

        y_update = [(80 - ba_height)] * 2 + [0] * 2
        new_xy = (
            np.array(
                list(zip([0] * 4, y_update)),
            )
            + self.indicator_battery_side_xy
        )
        self.indicator_battery_side.set_xy(new_xy)

    def _update_figure(self):
        for i, obs_part in enumerate(self.obs_values.values()):
            # setting new data
            self.obs_lines[i][0].set_data(self.line_x, obs_part[-self.visible_steps :])

            # rescaling y axis
            # based on https://stackoverflow.com/a/7198623
            axs = self.obs_axs[i]
            axs.relim()
            axs.autoscale_view(True, True, True)

            self.score_text.set_text(f"Score: {self.reward:.2f}")
            if self.game_finished:
                self.score_text.set_text(f"Game finished.\nScore: {self.reward:.2f}")

            self._update_house_figure()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _launch_update_requesting_thread(self, change=None):
        # pylint: disable=unused-argument
        def work(widget):
            max_steps = widget.episode_len
            for _ in range(max_steps):
                if widget.game_finished or widget.pause_requested:
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
            self.pause_requested = True
            self.active_thread.join()
            self.pause_requested = False

        self.active_thread = self._launch_update_requesting_thread()

    def _process_pause_request(self, change=None):
        # pylint: disable=unused-argument
        if self.active_thread:
            self.pause_requested = True
            self.active_thread.join()
            self.pause_requested = False

    def _process_reset_request(self, change=None):
        # pylint: disable=unused-argument
        self._process_pause_request()
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

            action = self.control.value
            action = np.array([action], dtype=np.float32)

            # pylint: disable=unused-variable
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.add_obs({**observation, self.reward_label: reward})

            self.reward += reward

            if terminated or truncated:
                self.game_finished = True
                self.control.set_trait("disabled", True)

            self._update_figure()

    def reset(self):

        obs = self.env.reset()
        obs = {**obs, self.reward_label: np.array([0], dtype=np.float32)}
        self.obs_values = {
            key: [np.array([0], dtype=np.float32)] * self.visible_steps
            for key in obs.keys()
            if key != "time_step"
        }
        self.add_obs(obs)
        self.reward = 0
        self.game_finished = False
        self.control.set_trait("disabled", False)

    def add_obs(self, obs):
        for key in self.obs_values.keys():
            self.obs_values[key].append(obs[key])

    def _go_to_game(self, change=None):
        # pylint: disable=unused-argument
        # Note: Confusingly "none" does not show the widget,
        # None does show the widget.
        self.start_screen.layout.display = "none"
        self.main_app.layout.display = None

    def _go_to_start_screen(self, change=None):
        # pylint: disable=unused-argument
        self.pause_button.click()
        self.start_screen.layout.display = None
        self.main_app.layout.display = "none"

    def _setup_start_screen(self):
        self.begin_button = widgets.Button(
            description="Start game",
        )
        self.game_logo = widgets.Image(
            value=self.game_logo_img,
            format="png",
            width=150,
            layout={"margin": "30px"},
        )
        self.begin_button.on_click(self._go_to_game)
        self.start_screen = widgets.VBox(
            children=[
                self.game_logo,
                self.begin_button,
            ],
            layout={
                "align_items": "center",
            },
        )