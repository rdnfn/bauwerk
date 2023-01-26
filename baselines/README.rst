===================
Running baselines
===================

We use the popular `Garage package <https://github.com/rlworkgroup/garage>`_ for our baseline results. As garage is not longer very actively maintained, installing it can be a bit tricky. Here are some tips:

1. Use Python 3.7 -> anything above has thrown errors for us about packages not resolving.
2. Really ensure that mujoco is installed correctly (and on the correct path) (needed for testing).
3. You may need to install patchelf `as described here <https://github.com/openai/mujoco-py/issues/652>`_.
