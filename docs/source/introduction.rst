============
Introduction
============

Radio Astronomy Gains and Visibility Inspector [``ragavi``]
***********************************************************

`Ragavi`_ is a python based software built for visualisation of radio astronomy data reduction by-products such as gains as well as visibility data. As the name implies, it comprises two aspects, a gain plotter aliased by ``ragavi-gains`` or ``ragavi``, which may be used as a command line tool and  within a notebook environment (Jupyter), and a visibility plotter aliased by ``ragavi-vis``, which is **only available** as a command line tool.


Motivation
**********
The main motivation for ``ragavi`` is introduced an in interactivity aspect to radio astronomy oriented plots traditionally be produced as static images in formats such as PNG, JPEG and SVG. This is achieved though a python package known as `Bokeh`_ which has the capability of producing stand alone HTML based interactive plots which are JavaScript oriented.

The output interactive plots enable actions such as

    * Panning: moving through a plot "frame by frame"
    * Zooming: focusing on a smaller or larger area
    * Scaling: increasing plot dimension depending on available window size
    * Selection:

among others, without requiring any redraw actions or special software to view plots. All that is required to view a plot produced by ``ragavi`` is a *web-browser*.


Limitations
***********
``Ragavi`` is still under development.


Main dependencies for this project

* `Daskms`_
* `Bokeh`_
* `Datashader`_


.. _Daskms: https://xarray-ms.readthedocs.io/en/latest/
.. _Bokeh: https://bokeh.pydata.org/en/latest/index.html
.. _Datashader: http://datashader.org/
.. _Ragavi: https://github.com/ratt-ru/ragavi

