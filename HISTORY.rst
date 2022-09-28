=======
History
=======

0.2.2 (2022-09-00)
------------------

* Features

  * Add support for much slower speed in game widget.
  * Add benchmarks.
  * Add support for setting tasks in environment.

* Improvements

  * Add clock and day to game widget.
  * Design improvements for game widget.
  * Add evaluation module.


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
