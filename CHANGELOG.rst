0.5.2
-----
**ragavi-gains**

  - Return -g/--gaintype and -kx/--k-xaxis for backward compatibility


0.5.1
-----
**ragavi-gains**

  - Gives default x-axis to unknown tables
  - Fixes partially hidden widget box sizes
  - Changes default x-axis for D-tables to time


0.5.0
-----
**ragavi-vis**

  - Fix corr selector again

**ragavi-gains**

  - Remove the `--gaintype` argument
  - Adds polcal tables Kcross, Xf and Df to allowed plots
  - Changes `-kx/--k-xaxis` to `-x\--xaxis` to allow for arbitrary specification of desired x-axis
  - Adds `-y/ --yaxis` as an alternative to `-d/--doplot`
  - Forces the temp directory ragavi uses to be the same as where ragavi is being ran. Attempt to fix "Out of memory error"

  
0.4.3
-----
**ragavi-gains**
  
  - Fix bug in flagged data display
  - Fixes link between batch selection and other selections


0.4.2
-----
**ragavi-vis**
  
  - Fixes correlation selection using corr label bug
  - Fixes mismatched colour map labels for coloured plots
  - Improved appearance of colour bar
  - Prevents crash if all data in a plot is flagged
  - Iterated plots now not linked by default. Added argument ``-lp / --link-plots`` to allow plots to be linked. All plots thus have independent axes by default.
  

0.4.1
-----
**ragavi-gains**
  
  - Supports all ``matplotlib`` type static outputs
  - Generate additional static plot for interactive plots with more than 30,000 points
  - Multiple tables also supported with static plots
  - Adds option ``all`` to plot all amplitude, phase, real and imaginary plots in a single plot

**ragavi-vis**
  
  - Added colour bar to coloured iterated plots
  

0.4.0
-----
**general**

  - Added arguments ``--debug`` and ``-lf / --logfile`` to allow the enabling of debug messages and specification of desired log file names respectively
  - Added argument ``-v`` / ``--version`` to display the current ``ragavi`` version.
  - Some improvements in logging

**ragavi-gains**

  - Fixed issue #70
  - Fixed issue #69

**ragavi-vis**
  
  - Changed argument ``-nf / --no-flagged`` to ``-if / --include-flagged`` for clarity.
  - Better handle of RAM usage


0.3.7
-----
**ragavi-gains**

  - Fixed legend display bug
  - ``--gain-type`` argument is now optional
  - Static and interactive files can be generated simultaneously by specifying both ``--plotname`` and ``--htmlname``
  
**general**

  - Fixed bug in phase wrapping
  
0.3.6
-----
**ragavi-vis**
  
  - Fixed bug in correlation selection

0.3.5
-----
**ragavi-vis**
 
  - Added information on read the docs
  - Fix error due to all null plots

0.3.4
-----
**ragavi-gains**
  
  - Fixes data selection bugs

0.3.3
-----
- **ragavi-vis**

  - Both ``--colour-axis`` and ``--iter-axis`` can now be used simultaneously
  - Added ``--cols`` to specify number of columns when ``--iter-axis`` is supplied.
  - Allowed selection of correlations via respective labels
  - All valid y-axis names are also valid in the x-axis
  - Changed UVWave scale from Lambda to Kilo Lambda
  - Improved averaging speed
  - Added ``--ant`` for antenna selection

- **ragavi-gains**

  - Deprecated the use of the name ``ragavi``. Now uses ``ragavi-gains``
  - Slightly improved layout


0.3.2
-----
- **ragavi-vis**

  - Added ``--iter-axis`` argument to generate grid plots over the iteration axis
  - ``--colour-axis`` generates a single plot coloured by the specified axis
  - Added ``--canvas-width`` and ``--canvas-height`` option to set resulting image resolution
  - Improved RAM management
  - Available X and Y axes names can be specified in short forms like CASA


0.3.1
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
  - Changed time x-axis to UTC time
  - Added new download selected data button
  - All available times displayed for bandpass plots


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
- Introduced ``ragavi`` visibility plotter accessible by ``ragavi-vis``
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
- Re-organise selector panel
- Added title and axis label size selectors
- Add field symbols alongside field names on check-boxes
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
- Travis release on tag
- Now plotting Flux calibration tables
- Extra frequency axis for bandpass plot


0.0.2
-----
- Module importable
- Table parameter option
