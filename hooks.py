"""MkDocs hooks.

Windlass documents with doctests, so docstrings contain lines like::

    >>> tokenize_words("a b")
    ['a', 'b']

``mkdocs-autorefs`` scans rendered docstring text for ``[...]`` and reports each
one it cannot resolve as a cross-reference. Doctest *output* is not markdown and
was never a reference, so those warnings are false positives — but under
``--strict`` any warning fails the build, which would mean choosing between
doctested docstrings and a strict docs build.

This hook filters exactly those, and nothing else. A genuine broken reference —
one written as ``[text][target]`` — still fails the build, because it does not
match the pattern below.
"""

from __future__ import annotations

import logging
import re

#: A doctest-output false positive: autorefs reports the unresolved target, and
#: for real references that target is a dotted identifier. These are list and
#: dict literals, quoted strings, and numbers.
_DOCTEST_OUTPUT = re.compile(
    r"Could not find cross-reference target '(?:"
    r"-?\d+"  # 0, -1
    r"|'[^']*'(?:,\s*'[^']*')*"  # 'a', 'b'
    r"|\"[^\"]*\""  # "name"
    r"|.*\bfor\b.*\bin\b.*"  # a comprehension echoed from an example
    r")'"
)


class _DoctestOutputFilter(logging.Filter):
    """Drops autorefs warnings that are really doctest output."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _DOCTEST_OUTPUT.search(record.getMessage())


def on_config(config: object, **kwargs: object) -> object:
    """Install the filter once MkDocs has set its logging up.

    The filter goes on the root *handlers* rather than on a logger. A filter
    attached to a logger only sees records logged directly to it; autorefs logs
    to its own logger and the record propagates to root, where MkDocs' strict
    counter lives. Handler-level filters see propagated records; logger-level
    ones do not.

    Args:
        config: The MkDocs config, returned unchanged.
        **kwargs: Ignored.

    Returns:
        The config, as the hook contract requires.
    """
    filter_ = _DoctestOutputFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(filter_)
    for name in ("mkdocs", "mkdocs.plugins", "mkdocs_autorefs"):
        logger = logging.getLogger(name)
        logger.addFilter(filter_)
        for handler in logger.handlers:
            handler.addFilter(filter_)
    return config
