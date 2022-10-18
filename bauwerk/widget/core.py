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
import ipympl  # pylint: disable=unused-import
from loguru import logger


import bauwerk
from bauwerk.constants import (
    GYM_NEW_RESET_API_ACTIVE,
    GYM_NEW_STEP_API_ACTIVE,
    GYM_RESET_INFO_DEFAULT,
)
import bauwerk.utils.data
import bauwerk.envs.solar_battery_house


class Game(widgets.VBox):
    """Bauwerk building control game widget."""

    def __init__(
        self,
        env: gym.Env = None,
        log_level: str = "error",
        height: int = 500,
        step_time=0.5,
        time_step_len=1 / 12,
        automatic_stepping=True,
        visible_h=24,
        episode_len=12 * 24 * 2,
        debug_mode=False,
        score_currency="€",
        score_scale=10.0,
        include_clock=True,
        alternative_plotting=True,
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

        # Set params
        self.automatic_stepping = automatic_stepping
        self.step_time = step_time
        self.fig_height = height - 150
        self.visible_steps = int(visible_h / time_step_len)
        self.reward_label = "reward (payment)"
        self.cfg = bauwerk.envs.solar_battery_house.EnvConfig(
            episode_len=episode_len,
            time_step_len=time_step_len,
        )
        self.debug_mode = debug_mode
        self.score_currency = score_currency
        self.include_clock = include_clock
        self.alternative_plotting = alternative_plotting

        # Apply scaling factor
        self.cfg.grid_peak_price *= score_scale
        self.cfg.grid_base_price *= score_scale
        self.cfg.grid_sell_price *= score_scale

        # Set up menu screens
        self.game_logo_img = bauwerk.utils.data.access_package_data(
            "assets/bauwerk_game_logo.png", None
        )
        self.buttons = {}
        self.menu_buttons = self._setup_menu_buttons()
        self._setup_start_screen()
        self._setup_settings_screen()

        if env is not None:
            self.custom_env = env
        else:
            self.custom_env = None
        self._set_env()

        self.active_thread = None
        self.pause_requested = False

        # Setting up controller
        action_high = self.env.action_space.high[0]
        action_low = self.env.action_space.low[0]

        self.control = widgets.FloatSlider(
            description="Charging",
            orientation="vertical",
            min=action_low,
            max=action_high,
            step=0.05,
            continuous_update=True,
            layout={"height": f"{self.fig_height}px"},
        )

        # Setting up main figure

        # This sets the first observations
        self.reset()
        self._setup_figure()

        self.game_lower_part = widgets.AppLayout(
            left_sidebar=self.control,
            center=self.fig.canvas,
            pane_widths=[1, 9, 0],
        )
        self.main_app = widgets.VBox(
            [
                widgets.VBox(
                    [
                        self.menu_buttons,
                    ]
                ),
                self.game_lower_part,
            ]
        )

        self.main_app.layout.display = "none"
        self.settings_screen.layout.display = "none"

        super().__init__(
            children=[
                self.start_screen,
                self.main_app,
                self.settings_screen,
            ],
            layout={
                "height": f"{height}px",
                "align_items": "center",
            },
        )

        self.game_finished = False

        # Setup automatic stepping
        if self.automatic_stepping:
            self.add_traits(step_requested=traitlets.Bool().tag(sync=True))
            self.step_requested = False
            self.observe(
                self._process_step_request, names="step_requested", type="change"
            )
            # self._launch_update_requesting_thread()
        else:
            self.control.observe(self.step, names="value")

            # disable start and pause buttons
            tooltip = (
                "DEACTIVATED because you are in manual mode."
                " Change control value to step forward."
            )
            for button in [self.buttons["start"], self.buttons["pause"]]:
                button.disabled = True
                button.tooltip = tooltip

    def _setup_menu_buttons(self):
        # Setting up menu
        self.buttons["start"] = widgets.Button(
            description="Start",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Start game",
            icon="play",  # (FontAwesome names without the `fa-` prefix)
        )
        self.buttons["start"].on_click(self._process_start_request)

        self.buttons["pause"] = widgets.Button(
            description="Pause",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Pause game",
            icon="pause",  # (FontAwesome names without the `fa-` prefix)
        )
        self.buttons["pause"].on_click(self._process_pause_request)

        self.buttons["reset"] = widgets.Button(
            description="Reset",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Reset game.",
            icon="refresh",  # (FontAwesome names without the `fa-` prefix)
        )
        self.buttons["reset"].on_click(self._process_reset_request)

        self.buttons["back_to_menu"] = widgets.Button(
            description="Menu",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Back to main menu.",
            icon="bars",  # (FontAwesome names without the `fa-` prefix)
        )
        self.buttons["back_to_menu"].on_click(self._go_to_start_screen)

        self.game_logo_small = widgets.Image(
            value=self.game_logo_img,
            format="png",
            width=50,
            layout={"margin": "0px 10px 0px 0px"},
        )

        return widgets.HBox(
            children=[
                self.game_logo_small,
                self.buttons["start"],
                self.buttons["pause"],
                self.buttons["reset"],
                self.buttons["back_to_menu"],
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

            plt.rcParams["font.family"] = "monospace"
            # plt.rcParams["font.weight"] = "bold"

            self.fig = plt.figure(
                constrained_layout=True,
                figsize=(7, fig_height),  # dpi=50
            )
            self.fig.canvas.header_visible = False
            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.resizable = False

            if self.debug_mode:
                self.fig.canvas.footer_visible = True
            else:
                self.fig.canvas.footer_visible = False

            plt.rcParams.update({"font.size": 10})

            # split canvas into left and right subfigures
            subfigs = self.fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 2])

            # Left handside of plt animation
            axs_left = subfigs[0].subplots(2, gridspec_kw={"height_ratios": [2, 1]})
            self._create_house_figure(axs_left[0])

            # Draw text
            axs_left[1].axis("off")
            self.score_text = axs_left[1].text(
                x=0.1,
                y=0.5,
                s=f"Score: 0.00{self.score_currency}",
                # animated=True,
                fontfamily="monospace",
                fontsize=15,
                fontweight="bold",
                color="white",
                bbox={"boxstyle": "Round", "facecolor": "black", "linewidth": 2.5},
            )

            # Right handside of plt animation
            # Create observation data plots

            self.line_x = np.linspace(0, self.visible_steps, self.visible_steps)

            if not self.alternative_plotting:
                self.obs_lines = []
                self.obs_axs = subfigs[1].subplots(len(self.obs_values))

                for i, (obs_name, obs_part) in enumerate(self.obs_values.items()):
                    self.obs_lines.append(
                        self.obs_axs[i].plot(
                            self.line_x,
                            obs_part[-self.visible_steps :],
                        )
                    )
                    self.obs_axs[i].set_title(obs_name.replace("_", " "))
            else:
                # plt.style.use("dark_background")
                # subfigs[1].set_facecolor("black")
                self.obs_axs = subfigs[1].subplots(3)
                self.obs_lines = {}
                self.obs_lines_fills = {}

                # adding pv gen and load to one plot
                self.obs_lines["load"] = self.obs_axs[0].plot(
                    self.line_x,
                    self.obs_values["load"][-self.visible_steps :],
                    color="red",
                )

                self.obs_lines["pv_gen"] = self.obs_axs[0].plot(
                    self.line_x,
                    self.obs_values["pv_gen"][-self.visible_steps :],
                    color="lightskyblue",
                )

                self.obs_axs[0].set_title("PV generation (blue) and load (red)")
                self.obs_axs[0].set_ylim(
                    -0.5, max(self.env.solar.max_value, self.env.load.max_value) + 0.5
                )

                # battery content plot
                self.obs_lines["battery_cont"] = self.obs_axs[1].plot(
                    self.line_x,
                    self.obs_values["battery_cont"][-self.visible_steps :],
                    color="white",
                )
                self.obs_axs[1].set_title("Battery content")
                self.obs_axs[1].set_ylim(-0.5, self.cfg.battery_size + 0.5)
                self.obs_lines_fills["battery_cont"] = self.obs_axs[1].fill_between(
                    self.line_x,
                    np.array(
                        self.obs_values["battery_cont"][-self.visible_steps :]
                    ).flatten(),
                    color="white",
                    alpha=0.5,
                )

                self.obs_lines[self.reward_label] = self.obs_axs[2].plot(
                    self.line_x,
                    self.obs_values[self.reward_label][-self.visible_steps :],
                )
                self.obs_axs[2].set_title(self.reward_label)

                for ax in self.obs_axs:
                    ax.set_facecolor("black")

            for ax in self.obs_axs:
                ax.label_outer()

    def _create_house_figure(self, img_ax):
        img_ax.axis("off")
        # "assets/house_v2.png"
        self.img_house = bauwerk.utils.data.access_package_data(
            "assets/house_v2.png",
            plt.imread,
        )
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

        clock_face = (1, 240 / 255, 195 / 255)

        if self.include_clock:

            self.time_day_text = img_ax.text(
                x=290,
                y=110,
                s="Day 1",
                # animated=True,
                fontfamily="monospace",
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox={"boxstyle": "Round", "facecolor": "black", "linewidth": 2.5},
            )

            self.time_text = img_ax.text(
                x=275,
                y=60,
                s="00:00",
                # animated=True,
                fontfamily="monospace",
                fontsize=14,
                fontweight="bold",
                color="black",
                bbox={"boxstyle": "Round", "facecolor": clock_face, "linewidth": 2.5},
            )

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

        if self.include_clock:
            days = self.current_step * self.cfg.time_step_len // 24 + 1
            hours = (self.current_step * self.cfg.time_step_len) % 24
            mins = (hours % 1) * 60
            self.time_text.set_text(f"{int(hours):02d}:{round(mins):02d}")
            self.time_day_text.set_text(f"Day {int(days)}")

    def _update_figure(self):
        if not self.alternative_plotting:
            for i, obs_part in enumerate(self.obs_values.values()):
                # setting new data
                self.obs_lines[i][0].set_data(
                    self.line_x, obs_part[-self.visible_steps :]
                )

                # rescaling y axis
                # based on https://stackoverflow.com/a/7198623
                axs = self.obs_axs[i]
                axs.relim()
                axs.autoscale_view(True, True, True)
        else:
            for key, value in self.obs_lines.items():
                value[0].set_data(
                    self.line_x, self.obs_values[key][-self.visible_steps :]
                )

            # update battery content fill below curve
            self.obs_lines_fills["battery_cont"].remove()
            self.obs_lines_fills["battery_cont"] = self.obs_axs[1].fill_between(
                self.line_x,
                np.array(
                    self.obs_values["battery_cont"][-self.visible_steps :]
                ).flatten(),
                color="white",
                alpha=0.5,
            )

            # rescale reward
            axs = self.obs_axs[2]
            axs.relim()
            axs.autoscale_view(True, True, True)

        self.score_text.set_text(f"Score: {self.reward:.2f}{self.score_currency}")

        if self.game_finished:
            self.score_text.set_text(
                f"Game finished\nScore: {self.reward:.2f}" f"{self.score_currency}"
            )
            self.score_text.set_backgroundcolor("darkolivegreen")

        self._update_house_figure()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _launch_update_requesting_thread(self, change=None):
        # pylint: disable=unused-argument
        def work(widget):
            max_steps = widget.cfg.episode_len
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
            action = np.array([action], dtype=self.env.cfg.dtype)

            # pylint: disable=unused-variable
            # Note: using old step API to ensure compatibility
            if GYM_NEW_STEP_API_ACTIVE:
                observation, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
            else:
                observation, reward, done, _ = self.env.step(action)

            self.add_obs({**observation, self.reward_label: reward})

            self.reward += reward
            self.current_step += 1

            if done:
                self.game_finished = True
                self.control.set_trait("disabled", True)

            self._update_figure()

    def reset(self):

        if GYM_NEW_RESET_API_ACTIVE and GYM_RESET_INFO_DEFAULT:
            obs, _ = self.env.reset()
        else:
            obs = self.env.reset()

        obs = {**obs, self.reward_label: np.array([0], dtype=float)}
        self.obs_values = {
            key: [np.array([0], dtype=float)] * self.visible_steps
            for key in obs.keys()
            if key not in ["time_step", "time_of_day"]
        }
        self.add_obs(obs)
        self.reward = 0
        self.game_finished = False
        self.current_step = 0

        if hasattr(self, "score_text"):
            # changing back the score text to black
            # on white background
            self.score_text.set_backgroundcolor("black")

        if hasattr(self, "control"):
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
        self.buttons["pause"].click()
        self.start_screen.layout.display = None
        self.settings_screen.layout.display = "none"
        self.main_app.layout.display = "none"

    def _go_to_settings_screen(self, change=None):
        # pylint: disable=unused-argument
        self.start_screen.layout.display = "none"
        self.main_app.layout.display = "none"
        self.settings_screen.layout.display = None

    def _setup_start_screen(self):
        self.game_logo = widgets.Image(
            value=self.game_logo_img,
            format="png",
            width=150,
            layout={"margin": "30px"},
        )
        self.buttons["go_to_game"] = widgets.Button(
            description="Start game",
        )
        self.buttons["go_to_game"].on_click(self._go_to_game)
        self.buttons["settings"] = widgets.Button(
            description="Settings",
        )
        self.buttons["settings"].on_click(self._go_to_settings_screen)
        self.start_screen = widgets.VBox(
            children=[
                self.game_logo,
                self.buttons["go_to_game"],
                self.buttons["settings"],
            ],
            layout={
                "align_items": "center",
            },
        )

    def _setup_settings_screen(self):
        self.settings_heading = widgets.HTML(
            value=(
                "<code style='color: black'>"
                "<h3 style='display: inline'>Settings</h3></code>"
            ),
        )
        self.settings = {}
        self.settings["battery_size"] = widgets.FloatSlider(
            description="Battery size (kWh)",
            orientation="horizontal",
            value=self.cfg.battery_size,
            min=0.1,
            max=30,
            step=0.1,
            continuous_update=False,
        )
        self.settings["episode_len"] = widgets.IntSlider(
            description="Episode length",
            orientation="horizontal",
            value=self.cfg.episode_len,
            min=2,
            max=24 * 31,
            step=1,
            continuous_update=False,
        )
        self.settings["step_time"] = widgets.FloatSlider(
            description="Step time (sec)",
            orientation="horizontal",
            value=self.step_time,
            min=0.001,
            max=5,
            step=0.001,
            continuous_update=False,
            tooltip="The higher, the slower the game.",
        )
        self.game_logo_settings = widgets.Image(
            value=self.game_logo_img,
            format="png",
            width=50,
            layout={"margin": "10px 0px 0px 0px"},
        )

        self.buttons["back_to_menu_from_settings"] = widgets.Button(
            description="Back",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Save changes and go back.",
        )

        def go_back(change=None):
            # pylint: disable=unused-argument
            self._set_env()
            self._go_to_start_screen()

        self.buttons["back_to_menu_from_settings"].on_click(go_back)

        self.settings_screen = widgets.VBox(
            children=[
                self.game_logo_settings,
                self.settings_heading,
                *self.settings.values(),
                self.buttons["back_to_menu_from_settings"],
            ],
            layout={
                "align_items": "center",
            },
        )

        for setting_slider in self.settings.values():
            setting_slider.layout.width = "500px"
            setting_slider.style.description_width = "200px"

    def _set_env(self):
        # Get cfg values from widgets
        self.cfg.battery_size = self.settings["battery_size"].value
        self.cfg.episode_len = self.settings["episode_len"].value
        self.step_time = self.settings["step_time"].value

        if self.custom_env is None:
            # Create underlying env if None given
            self.env = gym.make("bauwerk/SolarBatteryHouse-v0", cfg=self.cfg)
        else:
            self.env = self.custom_env

        self.reset()
