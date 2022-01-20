# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import io, os.path, re
from setuptools import setup, find_packages


def readfile(*parts):
    """Return contents of file with path relative to script directory"""
    herepath = os.path.abspath(os.path.dirname(__file__))
    fullpath = os.path.join(herepath, *parts)
    with io.open(fullpath, 'r') as f:
        return f.read()


def extract_varval(varname):
    """Extract value of __version__ variable by parsing python script"""
    initfile = readfile('src', 'sparse_ir', '__init__.py')
    var_re = re.compile(rf"(?m)^{varname}\s*=\s*['\"]([^'\"]*)['\"]")
    match = var_re.search(initfile)
    return match.group(1)


REPO_URL = "https://github.com/SpM-lab/sparse-ir"
VERSION = extract_varval('__version__')
MIN_XPREC_VERSION = extract_varval('min_xprec_version')
LONG_DESCRIPTION = readfile('README.rst')

setup(
    name='sparse-ir',
    version=VERSION,

    description=
        'intermediate representation (IR) basis for electronic propagator',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/x-rst',
    keywords=' '.join([
        'irbasis'
        ]),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        ],

    url=REPO_URL,
    author=[
        'Markus Wallerberger',
        'Hiroshi Shinaoka',
        'Kazuyoshi Yoshimi',
        'Junya Otsuki',
        'Chikano Naoya'
        ],
    author_email='markus.wallerberger@tuwien.ac.at',

    python_requires='>=3',
    install_requires=[
        'numpy',
        'scipy',
        'setuptools'
    ],
    extras_require={
        'test': ['pytest', 'irbasis', 'xprec'],
        'doc': ['sphinx>=2.1', 'sphinx_rtd_theme'],
        'xprec': [f'xprec>={MIN_XPREC_VERSION}'],
        },

    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    )
