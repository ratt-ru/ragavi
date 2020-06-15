# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime
from ragavi import ragavi, visibilities, utils, __version__

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
date = datetime.now()
project = 'ragavi'
copyright = f"{date.year}, Lexy Andati"
author = 'Lexy Andati'


# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage',
              'sphinx.ext.napoleon', 'sphinx.ext.intersphinx']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# pip install sphinx-rtd-theme
# pip install sphinx-press-theme 0.3.0

html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Personal settings ----- -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False

#html_logo = "./ragavi.png"
#html_favicon = "./favicon.ico"

"""
intersphinx_mapping = {
    'pypi': 'https://pypi.org/project/ragavi/',
    'ragavi': 'https://github.com/ratt-ru/ragavi/tree/visibilities',
    'bokeh': 'https://bokeh.pydata.org/en/latest/index.html',
    'datashader': 'http://datashader.org/',
    'dask-ms': 'https://xarray-ms.readthedocs.io/en/latest/api.html',
}

"""
# Latex settings
latex_elements = {
    'extraclassoptions': 'openany, oneside'
}
latex_show_urls = 'footnote'

autoclass_content = "class"

master_doc = 'index'
