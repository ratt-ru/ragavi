0.2.4
-----
- All argument parsers moved to ``arguments.py``

- **ragavi-vis**

  - Introduced MS averaging in ``ragavi-vis``
  - ``--cbin`` and ``--tbin`` added for channel and time averaging
  - ``--mem-limit`` and ``--num-cores`` for specifying memory limit per core and number of cores dask should use
  - Remove ``--image-name`` argument from ``ragavi-vis``

- **ragavi-gains**
  - Fixed field, correlation selection bugs #50
  - Fixed spectral window selection bug
  - Added spectral window selection widgets
  - Moved stats from plot titles to table below the plots


0.2.3
-----
- Add option `-kx` , `--k-xaxis` to allow selection of K table's x-axis (``ragavi-gains``)
- Values in `--field` can now be either comma or space separated


0.2.2
-----
- Add name of gain table plotted to the plot
- Delay (K) now plotted over time (Fixing #45)
- Fix bug with relative times (Fixing $46)


0.2.1
-----
- Fix some bugs with missing fields and correlations
- Only supporting python3 now


0.2.0
-----
- Introduced ragavi visibility plotter accessible by ragavi-vis
- Improved documentation
- Added progress bar for ``ragavi-vis``
- Changed gain plotter name to ``ragavi-gains``. Deprecating ``ragavi``
- Added ``--xmin``, ``--xmax``, ``--ymin``, ``--ymax`` options in `ragavi-vis` for selection of x and y data ranges
- Added ``--chunks`` command line option for user specified chunking strategies in ``ragavi-vis``
- Migrate from ``xarray-ms`` to ``dask-ms`` for table functions
- Added correlation selector on gain plots. All correlations plotted by default
- Removed ``--yu0, --yu1, --yl0, --yl1`` from `ragavi-gains`
- Fixed field selection and errorbar size bugs
- ``--field`` arguments in ``ragavi-gains`` **MUST** now be comma separated rather than space separated.


0.1.0
-----
- Error bars now have caps
- Introduced linked legends
- Default displayed data is now flagged
- Flagged data shown using inverted-triangle


0.0.9
-----
- Added flag button on plot
- Plotting D-Jones tables now supported
- Fixed bug in field_name to field_id converter


0.0.8
-----
- Fixed bug due to string encoding for python2.7


0.0.7
-----
- Updated version number


0.0.6
-----
- Now supporting python3
- All fields plotted by default on the same plot
- ``--field`` command line switch is now optional
- Different fields now plotted with different markers
- Migrated to ``xarray-ms`` from ``python-casacore``
- Added glyph alpha selector, glyph size selector, and field selector
- Reorganise selector panel
- Added title and axis label size selectors
- Add field symbols alongside field names on checkboxes
- Allow automatic plot scaling
- Medians now shown in plot titles


0.0.5
-----
- Added support for multiple table, fields and gaintype inputs
- Multiple table single field single gaintype input also allowed
- Plots from multiple tables plotted on single html file
- Added slider to change plot sizes
- All notifications and errors now logged to ragavi.log


0.0.4
-----
- Removed msname flag, Antenna names now show up in legends by default
- Support for string field names in addition to field indices
- Spectral window id, antenna name and scan id displayed on tooltip
- Remove second plot (for correlation 2) from delay table


0.0.3
-----
- Travis realease on tag
- Now plotting Flux callibration tables
- Extra frequency axis for bandpass plot


0.0.2
-----
- Module importable
- Table parameter option
