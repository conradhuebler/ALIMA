"""Search providers — side-effect imports register each provider on import.

Importing this package runs every ``@register_provider`` decorator, mirroring
``src/core/agents/__init__.py`` for steps/tool-fns. - Claude Generated
"""

from . import lobid as _lobid  # noqa: F401
from . import swb as _swb  # noqa: F401
from . import catalog as _catalog  # noqa: F401
from . import finc as _finc  # noqa: F401
from . import sru as _sru  # noqa: F401
from . import gnd_local as _gnd_local  # noqa: F401
from . import kvk as _kvk  # noqa: F401
