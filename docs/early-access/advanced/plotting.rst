========
Plotting
========

Bauwerk provides custom plotting functionality via the :class:`bauwerk.utils.plotting.EnvPlotter` class. This enables the plotting of agent trajectories. An example of using this functionality is given below:

.. code-block:: python

    import bauwerk
    import bauwerk.utils.plotting
    import gym

    env = gym.make("bauwerk/House-v0")
    initial_obs = env.reset() # assumes OpenAI Gym v0.21
    plotter = bauwerk.utils.plotting.EnvPlotter(
        initial_obs, env, visible_h=24
    )
    obs = initial_obs
    for _ in range(24):
        action = env.action_space.sample()
        step_return = env.step(action)
        plotter.add_step_data(action=action, step_return=step_return)

    plotter.update_figure()
    plotter.fig

Detailed plotting API
---------------------

.. autoclass:: bauwerk.utils.plotting.EnvPlotter
    :members: