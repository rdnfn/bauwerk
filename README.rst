.. raw:: html

   <p align="center">

.. image:: https://raw.githubusercontent.com/rdnfn/bauwerk/40684d5cd2ac70984f80670346dddb550d3b050a/docs/img/logo_v0.png
        :align: center
        :width: 200 px
        :alt: Logo

.. raw:: html

   </p>


.. image:: https://img.shields.io/pypi/v/bauwerk.svg
        :target: https://pypi.python.org/pypi/bauwerk

.. image:: https://readthedocs.org/projects/bauwerk/badge/?version=latest
        :target: https://bauwerk.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://mybinder.org/badge_logo.svg
        :target: https://mybinder.org/v2/gh/rdnfn/bauwerk/main?urlpath=voila/render/notebooks/demo.ipynb
        :alt: Interactive demo



Bauwerk is a *meta reinforcement learning* (meta RL) benchmark with building control environments. Bauwerk aims to facilitate the development of methods that *generalise* across buildings to help scale greener building controllers to more buildings.

**Motivation.** We use a lot of energy to operate buildings. About 30 percent of all human energy consumption are used for operating buildings. This energy runs a wide variety of devices with *heating, cooling, ventilation and air conditioning* (HVAC) systems accounting for more than half of that energy usage in many places. Shifting energy load away from carbon-intense grid energy has been suggested as a potential method to reduce the emission impact of buildings. Building control algorithms proposed in the literature often are too labour-intense during deployment to be commercially viable at scale.


.. _Game: https://mybinder.org/v2/gh/rdnfn/bauwerk/main?urlpath=voila/render/notebooks/demo.ipynb


Use cases
=========


.. list-table::
   :header-rows: 1
   :align: center

   * - 1. Meta RL benchmark
     - 2. Standard RL env
     - 3. Game
   * - .. image:: https://github.com/rdnfn/bauwerk/blob/29d3aea3cf93d5e11e2ae2b3d1b061661f4154ad/docs/assets/bauwerk_buildingcollection.png?raw=true
                :align: center
                :height: 50 px
                :alt: Meta RL illustration
     - .. image:: https://raw.githubusercontent.com/rdnfn/bauwerk/270d82236da39ab6e2c7c12bb400ff6f1fadada3/docs/assets/bauwerk_single_building.png
                :align: center
                :height: 120 px
                :alt: Meta RL illustration
     - .. image:: https://github.com/rdnfn/bauwerk/blob/2ca459ae9c286f6c01fb97007f4974400af9b047/docs/assets/game.gif?raw=true
                :align: center
                :height: 120 px
                :alt: Bauwerk game animation
   * - Bauwerk provides a new benchmark to evaluate the suitability of meta RL methods for building control. The diversity of buildings provides a challenging meta RL problem with potential to help towards real-world positive impact.
     - Even if you do not care (yet) about meta RL: Bauwerk provides a simple ``Gym`` environment for building control. Bauwerk's built-in python-based simulation makes it simpler to install than many other building frameworks: a ``pip install`` command is all you should need. A great way to get started in this space!
     - yes, Bauwerk also comes with a little (hacky) game for humans. Helps gain some intuition about otherwise abstract building control problems. `Try it out here. <https://mybinder.org/v2/gh/rdnfn/bauwerk/main?urlpath=voila/render/notebooks/Bauwerk%20game.ipynb>`_



Environments
============

Each simulated building simply has a home battery that can be used to charge low-carbon grid energy. Buildings vary in a number of parameters, including for example the size of the battery and solar photovoltaic installation. Bauwerk aims to provide an initial evaluation of a method's ability to generalise in the context of building control, whilst leaving a thorough investigation of the method's ability to manage complex physical to other work.

Besides the efficient simulation, a key benefit of Bauwerk simulations is that we can compute the theoretical *optimal control* for them.


Documentation
=============

https://bauwerk.readthedocs.io


License
=======

MIT license