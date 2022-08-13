"""Main widget module."""

import ipywidgets as widgets
import bauwerk
import gym

import numpy as np

bauwerk.setup()


class Game(widgets.HBox):
    """Bauwerk building control game widget."""

    def __init__(self):

        self.env = gym.make(
            "bauwerk/SolarBatteryHouse-v0",
            new_step_api=True,
        )

        action_high = self.env.action_space.high[0]
        action_low = self.env.action_space.low[0]

        self.control = widgets.FloatSlider(
            orientation="vertical",
            min=action_low,
            max=action_high,
            continuous_update=False,
        )
        self.control.observe(self.step, names="value")
        self.out = widgets.Output(layout={"border": "1px solid black"})

        with self.out:
            print("Get ready!")

        children = [self.control, self.out]
        super().__init__(children=children)

        self.game_finished = False
        self.reset()

    def reset(self):

        obs = self.env.reset()
        self.obs = [obs.values]

    def plot(self):
        pass

    def step(self, change):
        # pylint: disable=unused-argument

        if not self.game_finished:

            self.out.clear_output()

            action = self.control.value
            with self.out:
                print("Action", action)
            action = np.array([action], dtype=np.float32)
            # pylint: disable=unused-variable
            observation, reward, terminated, truncated, info = self.env.step(action)

            with self.out:
                print(observation)

                if terminated or truncated:
                    self.game_finished = True
                    self.control.set_trait("disabled", True)
                    print("Game over.")
