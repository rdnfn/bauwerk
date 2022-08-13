=======
History
=======

0.1.0 (2022-08-12)
------------------

* Improvements:

  * Add explicit CVXPY-based solver in ``bauwerk.envs.solvers`` for ``SolarBatteryHouseEnv`` that was missing earlier.
  * Update ``SolarBatteryHouseEnv`` to comply with new gym step API by returning ``truncated`` value.


0.1.0 (2022-08-12)
------------------

* First release on PyPI.

* Features:

  * ``SolarBatteryHouseEnv``: a simple environment representing a single family home with a solar photovoltaic installations and a home battery that can be controlled.
  * Optional integration with CVXPY to compute optimal actions in Bauwerk environments.
