# Copyright (C) 2020-2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import io, os.path, re
from setuptools import setup, find_packages
import versioneer


def readfile(*parts):
    """Return contents of file with path relative to script directory"""
    herepath = os.path.abspath(os.path.dirname(__file__))
    fullpath = os.path.join(herepath, *parts)
    with io.open(fullpath, 'r') as f:
        return f.read()


def rebase_links(text, base_url):
    """Rebase links to doc/ directory to ensure they work online."""
    doclink_re = re.compile(
                        r"(?m)^\s*\[\s*([^\]\n\r]+)\s*\]:\s*(doc/[./\w]+)\s*$")
    result, nsub = doclink_re.subn(r"[\1]: %s/\2" % base_url, text)
    return result


VERSION = versioneer.get_version()
git_hash = versioneer.get_versions()['full-revisionid']
REPO_URL = "https://github.com/SpM-lab/irbasis3"
DOCTREE_URL = "%s/tree/%s" % (REPO_URL, git_hash)
LONG_DESCRIPTION = rebase_links(readfile('README.md'), DOCTREE_URL)

setup(
    name='irbasis3',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

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
        'scipy'
    ],
    extras_require={
        'test': ['pytest', 'irbasis', 'xprec'],
        'doc': ['sphinx>=2.1', 'myst-parser', 'sphinx_rtd_theme'],
        },

    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    )
