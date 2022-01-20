import os

HEREPATH = os.path.abspath(os.path.dirname(__file__))
SRCDIR = os.path.join(HEREPATH, os.path.pardir, "src", "sparse_ir")


def check_whitespace(files):
    errors = []
    for fname in files:
        with open(fname, "r") as file:
            for lineno, line in enumerate(file, start=1):
                def add_error(msg):
                    errors.append((fname, lineno, line, msg))
                if not line:
                    break
                if line[-1] != '\n':
                    add_error("file must end in blank line")
                line = line[:-1]
                if line[-1:] == '\r':
                    add_error("file must only have unix line eendings")
                if line[-1:] == ' ':
                    add_error("line ends in whitespace")
                if '\t' in line:
                    add_error("line contains tab characters")
                if len(line) > 90:
                    add_error("line is too long: {:d} chars".format(len(line)))

    msg = ""
    for fname, lineno, line, lmsg in errors:
        msg += "{}:{}: {}\n".format(fname.name, lineno, lmsg)
    if msg:
        raise ValueError("Whitespace errors\n" + msg)


def python_files(path):
    for entry in os.scandir(path):
        if entry.is_file() and entry.name.endswith(".py"):
            yield entry


def test_ws_testdir():
    check_whitespace(python_files(HEREPATH))


def test_ws_srcdir():
    check_whitespace(python_files(SRCDIR))
