"""Make caught programming errors visible - Claude Generated (Aufräumen B).

A broad ``except Exception`` is the right tool for a failure the code EXPECTS:
the network is down, a JSON blob is malformed, an optional file is missing.
Continuing past those is the whole point.

It is the wrong tool for a failure that means *the code is wrong* — a method
that does not exist, a name that was never bound, a dependency that was never
installed. Those cannot be recovered from and will not fix themselves, but
caught by the same handler and logged at ``debug``/``warning`` they read exactly
like the expected kind.

That is not hypothetical. In July 2026 four separate defects survived for months
behind this exact ambiguity:

* ``update_gnd_entry`` — a method that does not exist, called from the GUI DNB
  sync; the AttributeError was caught and shown as "Fehler bei GND …", so the
  local GND store stayed empty and looked merely unused.
* ``_purge_pre_v2_swb_raw_rows`` — a dead migration (``rows[0][0]`` on a dict);
  the KeyError was logged at ``warning`` and the purge silently did nothing.
* ``rdflib`` — an undeclared dependency; the ImportError made every DNB lookup
  fail one id at a time, so a broken path looked like an empty one.
* ``save_as_provider_preferences`` — a NameError on every save (finding F-9).

The rule this module encodes: **keep catching everything, but say which kind it
was.** ``log_caught`` logs the bug-shaped exceptions at ERROR and everything else
at the level the caller considers normal. Nothing is re-raised — adopting it
cannot change control flow, only what shows up in the log.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

#: Exception types that indicate a defect rather than an unlucky runtime.
#:
#: ``TypeError`` is deliberately included: in this codebase it almost always
#: means a signature or shape mismatch. ``KeyError``/``IndexError``/``ValueError``
#: are deliberately EXCLUDED — they are routine for external data (a missing
#: field in an API response, an unparseable code) and flagging them would make
#: the signal worthless.
BUG_SHAPED_ERRORS = (AttributeError, NameError, ImportError, TypeError)


def is_bug_shaped(exc: BaseException) -> bool:
    """True when this exception means the code is wrong, not the world."""
    return isinstance(exc, BUG_SHAPED_ERRORS)


def log_caught(
    logger: Optional[logging.Logger],
    exc: BaseException,
    context: str,
    *,
    expected_level: str = "warning",
    detail: Any = None,
) -> bool:
    """Log an exception that was caught and swallowed. Never re-raises.

    Args:
        logger: destination; ``None`` is tolerated so call sites in objects that
            may not have one stay simple.
        exc: the caught exception.
        context: what was being attempted, in the caller's words.
        expected_level: level for the *expected* kind ("debug"/"info"/"warning").
        detail: optional extra, appended when present.

    Returns:
        True if it was logged as a defect (bug-shaped), else False — so a caller
        can additionally count or surface those without repeating the rule.
    """
    bug = is_bug_shaped(exc)
    suffix = f" [{detail}]" if detail else ""
    message = f"{context}: {type(exc).__name__}: {exc}{suffix}"
    if logger is None:
        return bug
    if bug:
        # Deliberately ERROR: this will not resolve on the next attempt, and the
        # whole point is that it stops looking like routine noise.
        logger.error(f"{message} — this looks like a defect, not a runtime failure")
    else:
        getattr(logger, expected_level, logger.warning)(message)
    return bug
