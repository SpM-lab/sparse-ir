# Copyright (C) 2020-2022 Markus Wallerberger, Hiroshi Shinaoka, and others
# SPDX-License-Identifier: MIT
import os

HEREPATH = os.path.abspath(os.path.dirname(__file__))
ROOTDIR = os.path.abspath(os.path.join(HEREPATH, os.path.pardir))
SRCDIR = os.path.join(ROOTDIR, "src", "sparse_ir")
DOCDIR = os.path.join(ROOTDIR, "doc")


def check_whitespace(files):
    errors = []
    blank = 0
    lineno = 0
    line = ""
    def add_error(fmt, *params):
        errors.append((fname, lineno, line, fmt.format(*params)))

    for fname in files:
        with open(fname, "r") as file:
            line = ""
            for lineno, line in enumerate(file, start=1):
                if line[-1:] != '\n':
                    add_error("file must end in blank line")
                line = line[:-1]
                if line:
                    blank = 0
                else:
                    blank += 1
                if line[-1:] == '\r':
                    add_error("file must only have unix line endings")
                if line[-1:] == ' ':
                    add_error("line ends in whitespace")
                if '\t' in line:
                    add_error("line contains tab characters")
                if len(line) > 90:
                    add_error("line is too long: {:d} chars", len(line))
            # end of file
            if blank != 0:
                add_error("file has {:d} superflouos blank lines", blank)

    msg = ""
    for fname, lineno, line, lmsg in errors:
        msg += "{}:{}: {}\n".format(fname.name, lineno, lmsg)
    if msg:
        raise ValueError("Whitespace errors\n" + msg)


def all_files(path, ext):
    for entry in os.scandir(path):
        if entry.is_file() and entry.name.endswith(ext):
            yield entry


def test_ws_testdir():
    check_whitespace(all_files(HEREPATH, ".py"))


def test_ws_srcdir():
    check_whitespace(all_files(SRCDIR, ".py"))


def test_ws_setup():
    check_whitespace(all_files(ROOTDIR, ".py"))


def test_ws_doc():
    check_whitespace(all_files(DOCDIR, ".rst"))
