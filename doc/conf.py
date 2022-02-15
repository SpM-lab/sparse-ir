# Configuration file for the Sphinx documentation builder.
# Run build as:
#   sphinx-build -M html . build
import io
import os
import re


def extract_varval(*varnames):
    """Return contents of file with path relative to script directory"""
    herepath = os.path.abspath(os.path.dirname(__file__))
    fullpath = os.path.join(herepath, '..', 'src', 'sparse_ir', '__init__.py')
    with io.open(fullpath, 'r') as f:
        contents = f.read()
    for varname in varnames:
        var_re = re.compile(rf"(?m)^{varname}\s*=\s*['\"]([^'\"]*)['\"]")
        match = var_re.search(contents)
        yield match.group(1)


# === Project information

project = 'sparse-ir'
copyright, version = extract_varval('__copyright__', '__version__')
release = version
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
