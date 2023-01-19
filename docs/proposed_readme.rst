.. raw:: html

   <p align="center">

.. image:: https://raw.githubusercontent.com/rdnfn/bauwerk/40684d5cd2ac70984f80670346dddb550d3b050a/docs/img/logo_v0.png
        :align: center
        :width: 120 px
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

**Background.** We use a lot of energy to operate buildings. About 30 percent of all human energy consumption are used for operating buildings. This energy runs a wide variety of devices with *heating, cooling, ventilation and air conditioning* (HVAC) systems accounting for more than half of that energy usage in many places. Shifting energy load away from carbon-intense grid energy has been suggested as a potential method to reduce the emission impact of buildings. Building control algorithms proposed in the literature often are too labour-intense during deployment to be commercially viable at scale.

**Benchmark.** Bauwerk provides a standardised benchmark to evaluate the suitability of meta RL methods for building control. Each simulated building simply has a home battery that can be used to charge low-carbon grid energy. Buildings vary in a number of parameters, including for example the size of the battery and solar photovoltaic installation. Bauwerk aims to provide an initial evaluation of a method's ability to generalise in the context of building control, whilst leaving a thorough investigation of the method's ability to manage complex physical to other work.


.. _Game: https://mybinder.org/v2/gh/rdnfn/bauwerk/main?urlpath=voila/render/notebooks/demo.ipynb

Features
========

- **Diverse set of buildings:** Bauwerk provides access to large set of environments, each corresponding to different building with different parameters.
- **Efficient simulation:** Bauwerk's built-in building simulation is purposefully simple to allow for rapid evaluation.
- **Optimal control:** the building simulations are based on fully tractable optimisation problems. Therefore, you can compute the best possible policy and compare your algorithm to a rigorous benchmark.
- **Game:** building control problems can be unintuitive to humans. Bauwerk provides a game interface to the environment that allows for a more intuitive understanding of the problems. This game aims to help intuitively understand and explain the problems.


Documentation
=============

https://bauwerk.readthedocs.io


License
=======

MIT license




