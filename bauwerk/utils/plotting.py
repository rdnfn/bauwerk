"""Plotting utility functions."""

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import bauwerk
import bauwerk.utils.gym


def plot_optimal_actions(env: bauwerk.HouseEnv, max_num_acts=None):

    initial_obs = bauwerk.utils.gym.force_old_reset(env.reset())
    plotter = EnvPlotter(
        initial_obs, env, visible_h=max_num_acts / env.cfg.time_step_len
    )
    opt_acts, _ = bauwerk.solve(env)

    if max_num_acts is None:
        max_num_acts = len(opt_acts)

    for i in range(min(env.cfg.episode_len, max_num_acts)):
        act = opt_acts[i]
        step_return = env.step(act)
        plotter.add_step_data(action=act, step_return=step_return)

    plotter.update_figure()

    return plotter.fig


class EnvPlotter:
    """Plotting class for Bauwerk environments."""

    def __init__(
        self,
        initial_obs: dict,
        env: bauwerk.HouseEnv,
        visible_h=24,
        fig_height: int = 600,
        debug_mode: bool = False,
        include_house_figure: bool = False,
        alternative_plotting: bool = True,
        include_clock_in_house_figure: bool = True,
        score_currency: str = "€",
        background: Optional[str] = "white",
        plot_grid_threshold: bool = True,
        plot_actions: bool = True,
        rescale_action: bool = True,
        plot_optimal_acts: bool = True,
    ) -> None:
        """Plotting class for Bauwerk environments."""

        self.reward_label = "Reward (payment)"
        self.score_currency = score_currency
        self.visible_h = visible_h
        self.visible_steps = int(visible_h / env.cfg.time_step_len) + 1
        self.fig_height = fig_height
        self.debug_mode = debug_mode
        self.include_house_figure = include_house_figure
        self.alternative_plotting = alternative_plotting
        self.include_clock = include_clock_in_house_figure
        self.env = env
        self.background = background
        self.plot_grid_threshold = plot_grid_threshold
        self.plot_actions = plot_actions
        self.rescale_action = rescale_action
        self.plot_optimal_acts = plot_optimal_acts

        if self.plot_optimal_acts:
            self.optimal_acts = bauwerk.solve(env)[0]

        self.reset(initial_obs)
        self._set_up_figure()

    def _add_obs(self, obs):
        for key in self.obs_values.keys():
            self.obs_values[key].append(obs[key])

    def _set_up_figure(self) -> None:

        # Setting up main figure
        with plt.ioff():

            ### Setting up main plt.Figure()

            # Setting correct height in pixels
            # Conversion following setup described in:
            # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
            px = 1 / plt.rcParams["figure.dpi"]
            fig_height = self.fig_height * px * 1  # in inches

            plt.rcParams["font.family"] = "monospace"

            self.fig = plt.figure(
                constrained_layout=True,
                figsize=(7, fig_height),  # dpi=50
                facecolor=self.background,
            )
            self.fig.canvas.header_visible = False
            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.resizable = False

            if self.debug_mode:
                self.fig.canvas.footer_visible = True
            else:
                self.fig.canvas.footer_visible = False

            plt.rcParams.update({"font.size": 10})

            if self.include_house_figure:
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
                right_handside = subfigs[1]
            else:
                right_handside = self.fig

            # Right handside of plt animation
            # (or main figure if not house figure)
            # This create observation data plots

            self.line_x = np.linspace(0, self.visible_h, self.visible_steps)

            if not self.alternative_plotting:
                self.obs_lines = []
                self.obs_axs = right_handside.subplots(len(self.obs_values))

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
                num_subplots = 3
                if self.plot_actions:
                    num_subplots += 1

                self.obs_axs = right_handside.subplots(num_subplots, sharex=True)
                self.obs_lines = {}
                self.obs_lines_fills = {}

                # adding pv gen and load to one plot
                self.obs_lines["info_load"] = self.obs_axs[0].plot(
                    self.line_x,
                    self.obs_values["info_load"][-self.visible_steps :],
                    color="red",
                )

                self.obs_lines["net_load"] = self.obs_axs[0].plot(
                    self.line_x[:-1],
                    self.obs_values["net_load"][-self.visible_steps + 1 :],
                    color="yellow",
                )

                self.obs_lines["info_pv_gen"] = self.obs_axs[0].plot(
                    self.line_x,
                    self.obs_values["info_pv_gen"][-self.visible_steps :],
                    color="lightskyblue",
                )
                self.obs_axs[0].hlines(
                    self.env.cfg.grid_peak_threshold,
                    0,
                    len(self.line_x),
                    label="No charging",
                    linestyle="--",
                    color="lightblue",
                )

                self.obs_axs[0].set_title(
                    (
                        "PV generation (blue), residential load (red)"
                        ", and net load (yellow)"
                    )
                )
                self.obs_axs[0].set_ylim(
                    -2.5, max(self.env.solar.max_value, self.env.load.max_value) + 1.0
                )
                self.obs_axs[0].set_ylabel("kW")

                # battery content plot
                self.obs_lines["info_battery_cont"] = self.obs_axs[1].plot(
                    self.line_x,
                    self.obs_values["info_battery_cont"][-self.visible_steps :],
                    color="white",
                )
                self.obs_axs[1].set_title("Battery content")
                self.obs_axs[1].set_ylim(-0.5, self.env.cfg.battery_size + 0.5)
                self.obs_lines_fills["info_battery_cont"] = self.obs_axs[
                    1
                ].fill_between(
                    self.line_x,
                    np.array(
                        self.obs_values["info_battery_cont"][-self.visible_steps :]
                    ).flatten(),
                    color="white",
                    alpha=0.5,
                )
                self.obs_axs[1].set_ylabel("kWh")

                if self.plot_actions:
                    self.obs_lines["charging_power"] = self.obs_axs[2].plot(
                        self.line_x[:-1],
                        self.obs_values["charging_power"][-self.visible_steps + 1 :],
                        color="red",
                    )
                    if self.plot_optimal_acts:
                        self.obs_lines["optimal_action"] = self.obs_axs[2].plot(
                            self.line_x[:-1],
                            self.obs_values["optimal_action"][
                                -self.visible_steps + 1 :
                            ],
                            color="white",
                            linestyle=(0, (1, 1)),
                            linewidth=2,
                            alpha=1,
                        )
                    self.obs_lines["action"] = self.obs_axs[2].plot(
                        self.line_x[:-1],
                        self.obs_values["action"][-self.visible_steps + 1 :],
                        color="white",
                    )

                    self.obs_axs[2].set_title("Control action (dotted line: optimal)")
                    if not self.rescale_action:
                        self.obs_axs[2].set_ylim(
                            -0.5 + self.env.action_space.low,
                            self.env.action_space.high + 0.5,
                        )
                    self.obs_axs[2].set_ylabel("Prop. of size")

                self.obs_lines[self.reward_label] = self.obs_axs[-1].plot(
                    self.line_x,
                    self.obs_values[self.reward_label][-self.visible_steps :],
                    color="lightgreen",
                    linestyle=(0, (1, 1)),
                )
                self.obs_lines["info_cost"] = self.obs_axs[-1].plot(
                    self.line_x,
                    self.obs_values["info_cost"][-self.visible_steps :],
                    color="lightgreen",
                )
                self.obs_axs[-1].set_title(
                    self.reward_label + " (dotted: includes penalty)"
                )
                self.obs_axs[-1].set_ylabel(self.score_currency)
                self.obs_axs[-1].set_xlabel("Time (h)")

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
            self.obs_values["info_pv_gen"][-1], self.env.solar, 0.2, 0.9
        )
        solar_ray_strength = get_alpha_value(
            self.obs_values["info_pv_gen"][-1], self.env.solar, 0.0, 0.5
        )
        self.indicator_solar.set_alpha(1 - solar_strength)
        self.indicator_solar_ray.set_alpha(solar_ray_strength)

        load_strength = get_alpha_value(
            self.obs_values["info_load"][-1], self.env.load, 0.4, 1
        )

        self.indicator_load.set_alpha(1 - load_strength)

        battery_cont = float(
            self.obs_values["info_battery_cont"][-1] / self.env.battery.size
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
            days = self.current_step * self.env.cfg.time_step_len // 24 + 1
            hours = (self.current_step * self.env.cfg.time_step_len) % 24
            mins = (hours % 1) * 60
            self.time_text.set_text(f"{int(hours):02d}:{round(mins):02d}")
            self.time_day_text.set_text(f"Day {int(days)}")

    def update_figure(self):
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
                if not key in [
                    "action",
                    "optimal_action",
                    "net_load",
                    "charging_power",
                ]:
                    value[0].set_data(
                        self.line_x, self.obs_values[key][-self.visible_steps :]
                    )
                else:
                    value[0].set_data(
                        self.line_x[:-1],
                        self.obs_values[key][-self.visible_steps + 1 :],
                    )

            # update battery content fill below curve
            self.obs_lines_fills["info_battery_cont"].remove()
            self.obs_lines_fills["info_battery_cont"] = self.obs_axs[1].fill_between(
                self.line_x,
                np.array(
                    self.obs_values["info_battery_cont"][-self.visible_steps :]
                ).flatten(),
                color="white",
                alpha=0.5,
            )

            # rescale reward
            axs = self.obs_axs[-1]
            axs.relim()
            axs.autoscale_view(True, True, True)

            if self.rescale_action:
                # rescale action
                axs = self.obs_axs[2]
                axs.relim()
                axs.autoscale_view(True, False, True)

        if hasattr(self, "score_text"):
            self.score_text.set_text(f"Score: {self.reward:.2f}{self.score_currency}")
            if self.game_finished:
                self.score_text.set_text(
                    f"Game finished\nScore: {self.reward:.2f}" f"{self.score_currency}"
                )
                self.score_text.set_backgroundcolor("darkolivegreen")

        if self.include_house_figure:
            self._update_house_figure()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def step(self, action, observation, reward):
        self._add_obs({**observation, self.reward_label: reward, "action": action})

        self.reward += reward
        self.current_step += 1

        self.update_figure()

    def add_step_data(self, step_return, action):
        observation = step_return[0]
        reward = step_return[1]
        info = step_return[-1]

        self._add_obs(
            {
                **observation,
                self.reward_label: reward,
                "action": action,
                "optimal_action": self.optimal_acts[info["time_step"] - 1],
                "net_load": info["net_load"],
                "charging_power": self.env.get_action_from_power(
                    info["charging_power"]
                ),
                "info_load": info["load"],
                "info_pv_gen": info["pv_gen"],
                "info_battery_cont": info["battery_cont"],
                "info_cost": -info["cost"],
            }
        )
        self.reward += reward
        self.current_step += 1

    def reset(self, obs):
        obs = {
            **obs,
            self.reward_label: np.array([0], dtype=float),
            "action": np.array([0], dtype=float),
            "optimal_action": np.array([0], dtype=float),
            "net_load": np.array([0], dtype=float),
            "charging_power": np.array([0], dtype=float),
            "info_pv_gen": np.array([0], dtype=float),
            "info_load": np.array([0], dtype=float),
            "info_battery_cont": np.array([0], dtype=float),
            "info_cost": np.array([0], dtype=float),
        }
        self.obs_values = {
            key: [np.array([0], dtype=float)] * (self.visible_steps + 1)
            for key in obs.keys()
            if key not in ["time_step", "time_of_day"]
        }
        self._add_obs(obs)
        self.reward = 0
        self.game_finished = False
        self.current_step = 0

        if hasattr(self, "score_text"):
            # changing back the score text to black
            # on white background
            self.score_text.set_backgroundcolor("black")
