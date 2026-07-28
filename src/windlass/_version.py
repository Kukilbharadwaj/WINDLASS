"""Single source of truth for the Windlass version.

The build backend reads ``__version__`` from this module, so it must stay a
plain string literal assignment with no imports or computation.
"""

__version__ = "0.1.1"
