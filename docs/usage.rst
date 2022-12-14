===============
Getting started
===============

Bauwerk provides `Gym`_ environments. Thus, Bauwerk can be used using the standard  `Gym`_ API as shown below:

.. code-block:: python

    import bauwerk
    import gym

    env = gym.make("bauwerk/SolarBatteryHouse-v0")
    obs = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()


.. admonition:: info

    The above example shows the `Gym`_ API version ``>=0.26``. Between `Gym`_ ``v0.21`` and ``v0.26`` this API has changed a lot. For example, ``env.step()`` now returns ``terminated`` and ``truncated`` instead of ``done``.

    **Bauwerk is compatible with all Gym API versions greater/equal v0.21**. To use a specific version of the `Gym`_ API, (after installing Bauwerk) run something like:

    .. code-block:: console

        pip install gym==0.26

    By satisfying older versions of this API Bauwerk can be used with popular RL libraries that may not have made the switch to the new API yet (such as `stable-baselines3`_).

.. _Gym: https://github.com/openai/gym
.. _stable-baselines3: https://github.com/DLR-RM/stable-baselines3

