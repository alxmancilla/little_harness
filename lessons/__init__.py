"""Offline learning labs; importing lessons performs no application setup."""

import sys

if sys.version_info < (3, 11):
    raise SystemExit("The labs need Python 3.11 or newer, but this is Python %s (%s). "
                     "See LEARN.md#before-you-start, or run with `uv run python -m lessons...`."
                     % (sys.version.split()[0], sys.executable))
