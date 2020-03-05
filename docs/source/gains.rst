``ragavi-gains``
================
To be used for gain table visualisation.

Currently, gain tables supported for visualisation by ``ragavi-gains`` are :

    * Flux calibration (F tables)
    * Bandpass calibration (B tables)
    * Delay calibration (D tables)
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
* While antenna data can be made visible through clicking its corresponding legend, this behaviour is not linked to the field, SPW, correlation selection checkboxes. Therefore, clicking the legend for a specific antenna will make data from all fields, SPWs and correlation for that antenna visible. As a workaround, data points can be identifies using tooltip information
* Unless **all** the available fields, SPWs and correlations have not been selected, the antenna legends will appear greyed-out. This is because a single legend is attached to multiple data for each of the available categories. Therefore, clicking on legends without meeting the preceeding condition may lead to some awkward results (a toggling effect).

To use ``ragavi-gains`` in a notebook environment, run in a notebook cell

.. code-block:: python

    from ragavi.ragavi import plot_table

    #specifying function arguments
    args = dict(mytabs=[], gain_types=[], cmap='viridis', doplot='ri', corr=1, ant='1,2,3,9,10')

    #inline plotting will be done
    plot_table(**args)

API
***
.. autoclass:: ragavi.ragavi.DataCoreProcessor
   :members: blackbox, act

.. autofunction:: ragavi.ragavi.plot_table
