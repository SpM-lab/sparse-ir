Hacking/Development
===================
For development, we recommend installing a "editable" version of the
repository with unit test dependencies::

    git checkout https://github.com/SpM-lab/sparse-ir.git
    cd sparse_ir
    pip install -e .[test]

Now the installed package automatically updates as you are working on the code.
Make sure you run unit tests::

    pytest


Code guidelines
---------------
The code tries to follow `PEP 8`_ more-or-less.

 1. Please note that lines are at most 79 characters!  In (occasionally)
    exceeding this limit, imagine yourself fighting through increasingly
    hostile terrain: at 83 characters, you are wading through smelly swamps, at
    90 characters, you are crawling through Mordor.

 2. Indentation is four spaces, no tabs.

 3. New code must have unit tests associated with it.

 4. Classes are abstractions.  If a class has no clear *thing* its modelling,
    it is not a class.

.. _`PEP 8`: https://www.python.org/dev/peps/pep-0008/
