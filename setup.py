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


def extract_version(*parts):
    """Extract value of __version__ variable by parsing python script"""
    initfile = readfile(*parts)
    version_re = re.compile(r"(?m)^__version__\s*=\s*['\"]([^'\"]*)['\"]")
    match = version_re.search(initfile)
    return match.group(1)


VERSION = extract_version('src', 'irbasis3', '__init__.py')
REPO_URL = "https://github.com/SpM-lab/irbasis3"
LONG_DESCRIPTION = readfile('README.md')

setup(
    name='irbasis3',
    version=VERSION,

    description=
        'intermediate representation (IR) basis for electronic propagator',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
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
        'scipy>=1.5' # For scipy.linalg.lapack.dgesjv
    ],
    extras_require={
        'test': ['pytest', 'irbasis', 'xprec'],
        'doc': ['sphinx>=2.1', 'myst-parser', 'sphinx_rtd_theme'],
        },

    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    )
