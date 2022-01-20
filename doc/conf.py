# Configuration file for the Sphinx documentation builder.
# Run build as:
#   sphinx-build -M html . build

# === Project information

project = 'sparse-ir'
copyright = '2021, Markus Wallerberger and others'
author = ', '.join([
    'Markus Wallerberger',
    'Hiroshi Shinaoka',
    'Kazuyoshi Yoshimi',
    'Junya Otsuki',
    'Chikano Naoya',
    ])

release = '0.2'
version = '0.2'

# === General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    ]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    }

#templates_path = ['_templates']
#html_static_path = ['_static']

exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    ]

html_theme = "sphinx_rtd_theme"
