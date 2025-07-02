# Configuration file for the Sphinx documentation builder.



import os
import sys
sys.path.insert(0, os.path.abspath('/Users/tancykao/Dropbox/05_Woolcock_DS/AnalyzeTools/TurtleWave-hdEEG'))  

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'TurtleWave hdEEG'
copyright = '2025, Tancy Kao'
author = 'Tancy Kao'
release = '2.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]



#templates_path = ['_templates']
#exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set theme
html_theme = 'sphinx_rtd_theme'


# Include Python modules in autodoc
#autodoc_member_order = 'bysource'
#autoclass_content = 'both'

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