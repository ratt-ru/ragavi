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
- --field command line switch is now optional
- Different fields now plotted with different markers
- Migrated to xarray-ms from python-casacore
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
