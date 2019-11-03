``ragavi-vis``
==============
To be used for visibility plotting. Supported arguments are

For x axis:

* amplitude
* antenna1
* antenna2
* frequency
* phase
* real
* scan
* time
* uvdistance
* uvwave

For y-axis:

* amplitude
* phase
* real
* imaginary

Iterations:

* Correlations (corr)
* Scan (scan)
* Spectral windows (spw)

Mandatory arguments are :code:`--xaxis`, :code:`--yaxis`, :code:`--table`.

API
***
.. autoclass:: ragavi.visibilities.DataCoreProcessor
   :members: blackbox, act

.. autofunction:: ragavi.visibilities.hv_plotter