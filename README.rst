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
    1. Bokeh
    2. Nodejs>=8
    3. Python-casacore

**- Install build dependencies:**

Click_ for instructions on how to install `python-casacore`.
Nodejs is a requirement for bokeh and can be installed using the commands

.. code-block:: bash
    
    $ sudo apt-get install curl
    $ curl -sL https://deb.nodesource.com/setup_8.x | bash -
    $ apt-get install -y nodejs

All python requirements are found in requirements.txt

or
 
To install nodejs in the virtual environment, use: nodeenv, a nodejs virtual environment.
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

.. _Click: https://github.com/casacore/python-casacore/blob/master/README.rst
.. _here: https://pypi.org/project/nodeenv
.. _source: https://github.com/ratt-ru/ragavi
.. _pep8: https://www.python.org/dev/peps/pep-0008
.. _license: https://github.com/ratt-ru/ragavi/blob/master/LICENSE
