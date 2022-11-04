=================
Experiment script
=================

Bauwerk provides an experiment script that allows us to evaluate SB3-based agents on Bauwerk distributions. This script can be used as a starting point for your experiments.

.. note::

    In order to run this script you will need to install the relevant dependencies using:

    .. code-block::

        pip install bauwerk[exp]

Usage
-----

There are two ways of using this script: either running ``bauwerk-exp`` or ``python -m bauwerk/exp/core.py`` (latter is only available in cloned Bauwerk repo).

So for example we can run:

.. code-block:: console

    bauwerk-exp sb3_alg=SAC train_steps_per_task=16000 env_cfg.battery_size=10

To get a full list of available configuration options, run

.. code-block:: console

    bauwerk-exp --help


Full script
-----------
.. literalinclude:: /../bauwerk/exp/core.py


