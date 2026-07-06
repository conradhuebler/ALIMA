"""Built-in-mode glue: registers CatalogProvider on package import - Claude Generated.

NOT executed when this dir is loaded as an external code plugin (the loader
imports only the manifest's entry module). Keep it a pure re-export.
"""

from .provider import CatalogProvider  # noqa: F401  (side-effect: @register_provider)
