``ragavi-vis``
==============
This is the visibility plotter. Supported arguments are

For x-axis:

* amplitude
* antenna1
* antenna2
* channel
* frequency
* imaginary
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

Iteration can also be activated through the :code:`-ia / --iter-axis` option and is possible over:

* antenna
* antenna1
* antenna2
* baseline
* corr (correlation)
* field
* Scan (scan)
* Spectral windows (spw)

It is also possible to colour over some axis (most of the iteration axes are supported) and this can be activated through the :code:`-ca / --colour-axis` argument. Please note that the mandatory arguments are :code:`--xaxis`, :code:`--yaxis`, :code:`--ms`.


Averaging
*********

``ragavi-vis`` has the ability to perform averaging before a plot is generated, by specifying the :code:`--cbin` or :code:`--tbin` arguments. These enable channel averaging and time averaging respectively. It is made possible through the use of the `Codex`_ africanus averaging API.

Averaging is performed over per SPW, FIELD and Scan. This means that data is grouped by DATA_DESC_ID, FIELD_ID and SCAN_NUMBER beforehand. However, data selection (such as selection of spws, fields, scans and baselines to be present) is done before the averaging to avoid processing data that is unnecessary for the plot. 


Resources
*********

Number of computer cores to be used, memory per core and the size of chunks to be loaded to each core may also be specified using :code:`-nc / --num-cores`, :code:`-ml / --memory-limit` and :code:`-cs / --chunk-size` respectively. These may play an active role in improving the performance and memory management as ``ragavi-vis`` runs. However, finding an optimal combination may be a tricky task but is well worth while. 

It is worth noting that supplying the x-axis and y-axis minimums and maximums may also significantly cut down the plotting time. This is because for minimum and maximum values to be calculated, ``ragavi-vis``' backends must pass through the entire dataset at least once before plotting begins and again as plotting continues, therefore, taking a longer time. While the effect of this may be minimal in small datasets, it is certainly amplified in large datasets.

.. _Codex: https://codex-africanus.readthedocs.io/en/latest/