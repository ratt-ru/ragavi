
======
ragavi
======

|Pypi Version|
|Build Version|
|Python Versions|

Radio Astronomy Gain and Visibility Inspector


============
Introduction
============

This library mainly requires
    1. `Bokeh`_
    2. `Python casacore`_
    3. `Daskms`_
    4. `Datashader`_
    5. Nodejs>=8

**- Install build dependencies:**

** `Python casacore`_ comes as a dependency of `Daskms`_ **
Nodejs is a requirement for ``Bokeh`` and can be installed using the commands

.. code-block:: bash
    
    $ sudo apt-get install curl
    $ curl -sL https://deb.nodesource.com/setup_8.x | bash -
    $ apt-get install -y nodejs

All python requirements are found in requirements.txt

or
 
To install ``nodejs`` in the virtual environment, use: ``nodeenv``, a nodejs virtual environment.
More info can be found here_

Create nodejs virtual environment with:

.. code-block:: bash
    
    $ nodeenv envName

and

.. code-block:: bash

    $ . envName/bin/activate

to switch to environment. 

============
Installation
============

Installation from source_,
working directory where source is checked out

.. code-block:: bash
  
    $ pip install .

This package is available on *PYPI* via

.. code-block:: bash
      
     $ pip install ragavi

=====
Usage
=====

Ragavi currently has two segements: 
  1. Gain plotter
  2. Visibility plotter

For the gain plotter, the name-space :code:`ragavi-vis` is used. To get help for this

.. note:: :code:`ragavi` namespace will soon change to  :code:`ragavi-vis`

.. code-block:: bash

    $ ragavi-gains -h

To use ragavi gain plotter

.. code-block:: bash

    $ ragavi-gains -t /path/to/your/table -g table_type (K / B/ F/ G/ D)

Multiple tables can be plotted on the same document simply by adding them in a space separated list to the :code:`-t` / :code:`--table` switch. 
They must however be accompanied by their respective gain table type in the :code:`-g` switch. e.g

.. code-block:: bash

    $ ragavi -t delay/table/1/ bandpass/table/2 flux/table/3 -g K B F


For the visibility plotter, the name-space :code:`ragavi-vis` is used. Help can be obtained by running

.. code-block:: bash

    $ ragavi-vis -h

To run ragavi-vis, the arguments :code:`--table`, :code:`--xaxis` and :code:`--yaxis` are basic requirements e.g.

.. code-block:: bash

    $ ragavi-vis --table /my/measurement/set --xaxis time --yaxis amplitude

=======
License
=======

This project is licensed under the MIT License - see license_ for details.

===========
Contribute
===========

Contributions are always welcome! Please ensure that you adhere to our coding standards pep8_.

.. |Pypi Version| image:: https://img.shields.io/pypi/v/ragavi.svg
                  :target: https://pypi.python.org/pypi/ragavi
                  :alt:
.. |Build Version| image:: https://api.travis-ci.com/ratt-ru/ragavi.svg?token=D5EL86dsmbhnuc9sNiRM&branch=master
                  :target: https://travis-ci.com/ratt-ru/ragavi
                  :alt:

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/ragavi.svg
                     :target: https://pypi.python.org/pypi/ragavi/
                     :alt:

.. _Python casacore: https://github.com/casacore/python-casacore/blob/master/README.rst
.. _here: https://pypi.org/project/nodeenv
.. _source: https://github.com/ratt-ru/ragavi
.. _pep8: https://www.python.org/dev/peps/pep-0008
.. _license: https://github.com/ratt-ru/ragavi/blob/master/LICENSE
.. _Bokeh: https://bokeh.pydata.org/en/latest/index.html
.. _Datashader: http://datashader.org/
.. _Daskms: https://xarray-ms.readthedocs.io/en/latest/