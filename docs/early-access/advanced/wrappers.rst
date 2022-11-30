======================
Wrappers
======================

Bauwerk provides a number of wrappers for its environments based on the `OpenAI Gym wrappers <https://www.gymlibrary.dev/api/wrappers/>`_. To use one of the wrappers, simply apply it to a Bauwerk environment. For example:

.. code-block:: python

    import bauwerk
    import bauwerk.envs.wrappers
    import gym

    env = gym.make("bauwerk/House-v0")
    wrapped_env = bauwerk.envs.wrappers.InfeasControlPenalty(env)


List of available wrappers
==========================

.. automodule:: bauwerk.envs.wrappers
    :members:

