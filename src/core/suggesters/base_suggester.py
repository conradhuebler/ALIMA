#!/usr/bin/env python3
# base_suggester.py

from typing import List, Dict, Any, Set, Optional, Union
from pathlib import Path
import sys
import json
from abc import ABC, ABCMeta, abstractmethod
from PyQt6.QtCore import QObject, pyqtSignal

class BaseSuggesterError(Exception):
    """Base exception for keyword suggestion errors"""
    pass

# Definiere eine neue Metaklasse, die von beiden erbt
class QObjectABCMeta(type(QObject), ABCMeta):
    pass


class BaseSuggester(QObject, ABC, metaclass=QObjectABCMeta):
    """
    Abstract base class for keyword suggestion systems.
    All suggester implementations should inherit from this class.
    """
    # Signal emitted when processing a search term
    currentTerm = pyqtSignal(str)
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None, debug: bool = False):
        """
        Initialize the suggester.
        
        Args:
            data_dir: Directory for data storage/caching
            debug: Whether to enable debug output
        """
        super().__init__()
        self.debug = debug
        self.data_dir = self._get_data_dir(data_dir)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise BaseSuggesterError(f"Could not create data directory: {str(e)}")
    
    def _get_data_dir(self, data_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Get the data directory path.
        
        Args:
            data_dir: Specified data directory or None
            
        Returns:
            Path to the data directory
        """
        if not data_dir:
            # Default to a subdirectory in the current script's directory
            return Path(sys.argv[0]).parent.resolve() / "data" / self.__class__.__name__.lower()
            
        if isinstance(data_dir, str):
            return Path(data_dir)
            
        if isinstance(data_dir, Path):
            return data_dir
            
        raise BaseSuggesterError(
            "Given data_dir is neither string nor Path. Cannot proceed"
        )
    
    @abstractmethod
    def prepare(self, force_download: bool = False) -> None:
        """
        Prepare the suggester by downloading or preparing necessary data.
        
        Args:
            force_download: Whether to force data download/preparation even if it's already available
        """
        pass
    
    @abstractmethod
    def search(self, terms: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Search for keywords based on the provided search terms.
        
        Args:
            terms: List of search terms
            
        Returns:
            Dictionary with structure:
            {
                search_term: {
                    keyword: {
                        "count": int,              # Number of occurrences
                        "gndid": set,              # Set of GND IDs
                        "ddc": set,                # Set of DDC classifications
                        "dk": set                  # Set of DK classifications
                    }
                }
            }
        """
        pass
    
    def get_unified_results(self, terms: List[str]) -> List[List]:
        """
        Get unified search results in the specified format.
        
        Args:
            terms: List of search terms
            
        Returns:
            List of results, where each result is a list with elements:
            [keyword, gnd_id, ddc, dk, count, search_term]
        """
        results = []
        search_results = self.search(terms)
        
        for term, keywords in search_results.items():
            for keyword, data in keywords.items():
                # Get the first GND ID if available (or empty string)
                gnd_id = next(iter(data.get("gndid", set())), "")
                
                # Get the first DDC classification if available (or empty string)
                ddc = next(iter(data.get("ddc", set())), "")
                
                # Get the first DK classification if available (or empty string)
                dk = next(iter(data.get("dk", set())), "")
                
                # Get the count (default to 1 if not available)
                count = data.get("count", 1)
                
                # Add the result
                results.append([keyword, gnd_id, ddc, dk, count, term])
                
        return results
