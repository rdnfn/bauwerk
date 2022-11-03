======================
Building distributions
======================

Besides the standard Gym environments, Bauwerk also provides distributions
over buildings.


API
---


For this purpose Bauwerk follows a simplified version of the API of the widely
used Meta-RL library `MetaWorld <https://github.com/rlworkgroup/metaworld>`_.

.. code-block:: python

    build_dist_b = bauwerk.benchmarks.BuildDistB()

    training_envs = []
    for task in build_dist_b.train_tasks:
        env = build_dist_b.make_env()
        env.set_task(task)
        training_envs.append(env)

    for env in training_envs:
        env.reset()
        act = env.action_space.sample()
        env.step(act)



Further, Benchmark also support the full
`MetaWorld <https://github.com/rlworkgroup/metaworld>`_ API. Note that this is for
compatibility reasons, and in many use-cases unnecessarily complex as all Bauwerk
Benchmarks only have a single environment class.

.. code-block:: python

    # Construct the benchmark, sampling tasks
    build_dist_b = bauwerk.benchmarks.BuildDistB()

    training_envs = []
    for name, env_cls in build_dist_b.train_classes.items():
        env = env_cls()
        task = random.choice(
            [task for task in build_dist_b.train_tasks if task.env_name == name]
        )
        env.set_task(task)
        training_envs.append(env)

    for env in training_envs:
        env.reset()
        act = env.action_space.sample()
        env.step(act)

Each distribution implements (and accept arguments of) the following class

.. autoclass:: bauwerk.benchmarks.BuildDist

Available distributions
-----------------------

Bauwerk provides a number of building distributions, each with increasing technical difficulty. The first and easiest distribution, A, always samples the same identical building. The next distribution, B, samples houses with different battery sizes, but keeps the house otherwise identical. The other distributions follow a similar pattern, each adding a bit more complexity to the problem. See the descriptions below for more details on each distribution.


.. autoclass:: bauwerk.benchmarks.BuildDistA
.. autoclass:: bauwerk.benchmarks.BuildDistB
.. autoclass:: bauwerk.benchmarks.BuildDistC
.. autoclass:: bauwerk.benchmarks.BuildDistD
.. autoclass:: bauwerk.benchmarks.BuildDistE