# Configuration file for the Sphinx documentation builder.
# Run build as:
#   sphinx-build -M html . build
import tomllib
import os

# Read version from pyproject.toml
def get_version():
    with open(os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml'), 'rb') as f:
        data = tomllib.load(f)
    return data['project']['version']

# === Project information

project = 'sparse-ir'
version = get_version()
release = version
copyright = '2024, SpM-lab'
author = ', '.join([
    'Markus Wallerberger',
    'Hiroshi Shinaoka',
    'Kazuyoshi Yoshimi',
    'Junya Otsuki',
    'Chikano Naoya',
    ])

# === General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
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
