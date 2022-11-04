.. highlight:: shell

============
Installation
============


Stable release
--------------

To install bauwerk, run this command in your terminal:

.. code-block:: console

    $ pip install bauwerk

This is the preferred method to install bauwerk, as it will always install the most recent stable release.

.. note::

    All Bauwerk dependencies will be automatically installed via pip, no further installation steps are required. Bauwerk does not rely on external building simulation engines such as EnergyPlus.

If you want to use some of the advanced features you can use the following pip install commands with extras to get the relevant dependencies:

- ``pip install bauwerk[opt]``: all optimisation code (e.g. if you want to use ``bauwerk.solve()``).
- ``pip install bauwerk[widget]``: anything to do with the game widget.
- ``pip install bauwerk[exp]``: for running experiments using the ``bauwerk-exp`` script.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for bauwerk can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/rdnfn/bauwerk

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/rdnfn/bauwerk/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/rdnfn/bauwerk
.. _tarball: https://github.com/rdnfn/bauwerk/tarball/master
