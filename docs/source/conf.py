# Configuration file for the Sphinx documentation builder.



import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Check if we're running on Read the Docs
on_rtd = os.environ.get('READTHEDOCS') == 'True'


# -- Project information -----------------------------------------------------
project = 'TurtleWave hdEEG'
copyright = '2025, Tancy Kao'
author = 'Tancy Kao'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',  # Add cross-references to other documentation
    'sphinx_rtd_theme', # Check documentation coverage
]

# Mock imports for C extensions or other hard-to-install dependencies
# This prevents build failures on Read the Docs when dependencies cannot be installed
autodoc_mock_imports = [
    'PyQt5', 
    'QtWidgets', 
    'QtCore', 
    'QtGui'
    # Add other C extensions or difficult dependencies here
]

# Master document (required for older versions of Sphinx on RTD)
master_doc = 'index'

# Exclude patterns for documentation build
exclude_patterns = [
    '_build', 
    'Thumbs.db', 
    '.DS_Store', 
    '**.ipynb_checkpoints'
]

# -- Options for HTML output -------------------------------------------------

# Set theme
html_theme = 'sphinx_rtd_theme'


# add theme options for Read the Docs
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# HTML options
html_static_path = ['_static']
html_logo = None  # Add path to your logo if you have one

# Autodoc settings
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Napoleon settings for NumPy and Google docstrings
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
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mappings for external references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}