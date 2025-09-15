"""App package exports.

Expose ``server`` and ``TRANSFORMS_DATA`` for backward compatibility with tests
and scripts that import these directly from ``app``. This will import the
bootstrap module which initializes data and registers routes, but it will NOT
start the web server process on import.
"""

# Importing bootstrap creates the Dash app and initializes data, without
# running the server. This preserves expected test semantics.
from .bootstrap import app, server  # noqa: F401
from etl.transforms import DATA as TRANSFORMS_DATA  # noqa: F401

__all__ = ["app", "server", "TRANSFORMS_DATA"]