"""Local subject-indexing database (own SQLite schema, ``schema.py``) as a lookup plugin.

Importing this package registers ``provider.GeoSachdbLookup`` as a built-in lookup
(``src/utils/lookups/__init__.py`` imports it). Without a usable database the
lookup reports ``is_available() == False`` and generates no tools. - Claude Generated"""

from . import provider  # noqa: F401 — registers the geo_sachdb lookup
