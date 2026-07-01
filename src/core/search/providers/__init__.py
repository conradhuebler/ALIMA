"""Search providers — side-effect imports register each provider on import.

Importing this package runs every ``@register_provider`` decorator, mirroring
``src/core/agents/__init__.py`` for steps/tool-fns. - Claude Generated
"""

from . import lobid_provider as _lobid  # noqa: F401
from . import swb_provider as _swb  # noqa: F401
from . import catalog_provider as _catalog  # noqa: F401
from . import finc_provider as _finc  # noqa: F401
from . import sru_provider as _sru  # noqa: F401
from . import gnd_local_provider as _gnd_local  # noqa: F401
