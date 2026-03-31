"""
ALIMA Catalog Clients - Claude Generated

Provides catalog search clients for different protocols:
- BiblioClient: Libero SOAP API (original)
- MarcXmlClient: MARC XML via SRU protocol (DNB, Library of Congress, etc.)
"""

try:
    from .biblio_client import BiblioClient
except ImportError:
    BiblioClient = None

from .marcxml_client import MarcXmlClient
from .rvk_api_client import RvkApiClient
from .rvk_marc_index import RvkMarcIndex

__all__ = ['BiblioClient', 'MarcXmlClient', 'RvkApiClient', 'RvkMarcIndex']
