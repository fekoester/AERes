import os
import sys
sys.path.insert(0, os.path.abspath('../../AERes/'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AERes'
copyright = '2024, Felix Köster'
author = 'Felix Köster'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Includes documentation from docstrings
    'sphinx.ext.coverage',  # Checks documentation coverage
    'sphinx.ext.napoleon',  # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
