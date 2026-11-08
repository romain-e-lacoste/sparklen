# Author: Romain E. Lacoste
# License: BSD-3-Clause

"""Source files compile without warnings (e.g. invalid escape sequences).

A backslash in a docstring that is not raw, such as LaTeX in a formula, is an
invalid escape sequence. Python reports it with a SyntaxWarning only when it
compiles the file, not when it loads a cached .pyc, so importing the package
does not catch it: each source file is compiled here with warnings as errors.
"""

import pathlib
import warnings

import pytest

import sparklen

# Skip __pycache__: the build script leaves empty __init__.py files there.
SOURCES = sorted(path for path in pathlib.Path(sparklen.__file__).parent.rglob("*.py")
                 if "__pycache__" not in path.parts)


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.name)
def test_source_compiles_without_warnings(path):
    """Check that the source file compiles without any warning,
    such as an invalid escape sequence in a LaTeX docstring."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
