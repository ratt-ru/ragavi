``ragavi-gains``
================
To be used for gain table visualisation.

Currently, gain tables supported for visualisation by ``ragavi-gains`` are :

    * Flux calibration (F tables)
    * Bandpass calibration (B tables)
    * Delay calibration (K tables)
    * Gain calibration (G tables)
    * D-Jones Leakage tables (D tables)

Mandatory fields are :code:`--table`, :code:`--gain_type`

If a field name is not specified to ``ragavi-vis`` all the fields will be plotted by default

It is possible to place multiple gain table plots of different [or same] types into a single HTML document using ``ragavi`` This can be done by specifying the table names and gain types as space separate list as below

.. code-block:: bash

    $ragavi-gains --table table/one/name table/two/name table/three/name table/four/name --gain_types B G D F --fields 0

This will yield an output HTML file, with plots in the order

+----------+--------+------+
| --table  | field  | gain | 
+----------+--------+------+
| table1   |   0    |    B |
+----------+--------+------+
| table2   |   0    |    G |
+----------+--------+------+
| table3   |   0    |    D | 
+----------+--------+------+
| table4   |   0    |    F |
+----------+--------+------+

Note
----
* At least a single field, spectral window and correlation **must** be selected in order for plots to show up.
* While antenna data can be made visible through clicking its corresponding legend, this behaviour is not linked to the field, SPW, correlation selection checkboxes. Therefore, clicking the legend for a specific antenna will make data from all fields, SPWs and correlation for that antenna visible. As a workaround, data points can be identified using tooltip information
* Unless **all** the available fields, SPWs and correlations have not been selected, the antenna legends will appear greyed-out. This is because a single legend is attached to multiple data for each of the available categories. Therefore, clicking on legends without meeting the preceeding condition may lead to some awkward results (a toggling effect).

To use ``ragavi-gains`` in a notebook environment, run in a notebook cell

.. code-block:: python

    from ragavi.ragavi import plot_table

    #specifying function arguments
    args = dict(mytabs=[], gain_types=[], cmap='viridis', doplot='ri', corr=1, ant='1,2,3,9,10')

    #inline plotting will be done
    plot_table(**args)

It is possible to generate PNG and SVG with ``ragavi-gains`` via two methods. The first method involves generating the HTML format first and then using the save tool found in the toolbar to download the plots. This method requires minimal effort although it may be a necessary redundancy to achieve the static image goal. The second method requires some additional setup, which may be slightly more taxing. 

**Step 1**: Install selenium, which can be done via:

.. code-block:: bash

    pip install selenium

**Step 2**: Download geckodriver for Firefox web browser, or chromedriver for Google chrome browser, and add it's executable to ``PATH``. Because Ubuntu linux ships with Firefox by default, I will demonstrate geckodriver v0.26.0, which is the latest driver at the time of wringing. The download page is `Geckodriver`_ 

.. code-block:: bash

    mkdir gecko && cd gecko

    wget https://github.com/mozilla/geckodriver/releases/download/v0.26.0/geckodriver-v0.26.0-linux32.tar.gz

    tar -xvzf geckodriver-v0.26.0-linux32.tar.gz

    chmod +x geckodriver

    export PATH=$PATH:/absolute/path/to/gecko

With this setup, one can now supply to ``ragavi-gains`` a ``--plotname`` value, which will result in the generation of PNG/SVG files, depending on the file extension provided. If, for example the plotname provided is ``foo.png``, ``ragavi-gains`` will assume the desired output should be PNG. The same applies for SVG.

It is necessary to point out that by default, ``ragavi`` uses the canvas image backend for interactive plots, due to performance issues associated with SVG image backend as stated in the `docs`_.
The default plots generated are always in HTML format.


API
***
.. autoclass:: ragavi.ragavi.DataCoreProcessor
   :members: blackbox, act

.. autofunction:: ragavi.ragavi.plot_table


.. _Geckodriver: https://github.com/mozilla/geckodriver/releases
.. _docs: https://docs.bokeh.org/en/latest/docs/user_guide/export.html#svg-generation