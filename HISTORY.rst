=======
History
=======


0.3.2 (2022-12-00)
------------------

* Improvements

  * Add information on evaluation to docs.

* Fixes

  * Improve plotting/widget code to comply with stricter shape testing in newer matplotlib versions

0.3.1 (2022-11-03)
------------------

* Features

  * New building distributions:

    * *Building Distribution C*: varies battery and solar installation size.
    * *Building Distribution D*: varies battery, solar installation and load consumption size/scale.
    * *Building distribution E*: varies same as above, and adds irreducible noise to load and solar traces.

* Environment & distribution changes (*! indicates that the change may affect experimental results*)

  * **!** Parameter default of ``grid_peak_threshold`` changed from 4.0kW to 2.0kW in all environments, including those of building distribution B.
  * The load and solar traces in ``HouseEnv`` now can be augmented with irreducible noise. This is set via the ``solar/load_noise_magnitude`` parameter of the ``EnvConfig``.

* Improvements

  * Update to docs on distributions, wrappers and more.

0.3.0 (2022-10-26)
------------------

* Features

  * Add experiment script
  * Add extensive evaluation features
  * Add support for much slower speed in game widget
  * Add benchmarks
  * Add support for setting tasks in environment
  * Add wrappers:

    *  that add task parameters to observation space.
    *  that clip the reward
    *  that clip the action space
    *  that normalise observation space

* Improvements

  * Add clock and day to game widget
  * Design improvements for game widget
  * Action spaces dtype can now be set in env cfg


0.2.1 (2022-09-12)
------------------

* Features

  * Add ``time_of_day`` variable to observation space.

* Improvements

  * The ``solve`` function is now directly imported with ``bauwerk``, to simplify usage.
  * Additional sections added to documentation.
  * Add more grid parameters to ``SolarBatteryHouseEnv`` configuration.

* Fixes:

  * Ensure solver outputs actions that are valid in environment (i.e. normalised)

0.2.0 (2022-09-09)
------------------

* Features:

  * Add game widget based on ``SolarBatteryHouseEnv``.

    * Includes browser-based version of Bauwerk game that can be played by anybody without installing anything.

  * Add support for selling to the grid in ``SolarBatteryHouseEnv``.

* Improvements:

  * Add explicit CVXPY-based solver in ``bauwerk.envs.solvers`` for ``SolarBatteryHouseEnv`` that was missing earlier.
  * Update ``SolarBatteryHouseEnv`` to comply with new gym step API by returning ``truncated`` value.
  * Add automatic github-actions-based testing of package.


0.1.0 (2022-08-12)
------------------

* First release on PyPI.

* Features:

  * ``SolarBatteryHouseEnv``: a simple environment representing a single family home with a solar photovoltaic installations and a home battery that can be controlled.
  * Optional integration with CVXPY to compute optimal actions in Bauwerk environments.
