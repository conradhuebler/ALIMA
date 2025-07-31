#!/usr/bin/env python3
"""
Unified URL/DOI Resolver - Enhanced web crawling with comprehensive content extraction
Claude Generated - Unifies DOI resolution logic with URL crawling and enhanced Springer support
"""

import re
import requests
import asyncio
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
import logging


class UnifiedResolver:
    """Unified URL/DOI resolution with enhanced Springer crawling and generic web support - Claude Generated"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def resolve(self, input_string: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Resolve URL or DOI to abstract text and metadata

        Args:
            input_string: URL, DOI, or DOI URL (e.g., "10.1007/...", "https://doi.org/10.1007/...", "https://link.springer.com/...")

        Returns:
            Tuple[success: bool, metadata: Dict, abstract_text: str]
        """
        try:
            self.logger.info(f"Resolving input: {input_string}")

            # Determine input type and extract DOI/URL
            input_type, resolved_value = self._analyze_input(input_string)

            if input_type == "springer_url":
                self.logger.info("Detected Springer URL, using enhanced web crawling")
                return self._resolve_springer_url(resolved_value)
            elif input_type == "springer_doi":
                self.logger.info("Detected Springer DOI, using enhanced web crawling")
                return self._resolve_springer_doi(resolved_value)
            elif input_type == "generic_url":
                self.logger.info("Detected generic URL, using web crawling")
                return self._resolve_generic_url(resolved_value)
            elif input_type == "crossref_doi":
                self.logger.info("Using CrossRef API for DOI resolution")
                return self._resolve_crossref_doi(resolved_value)
            else:
                error_msg = f"Unable to determine input type for: {input_string}"
                self.logger.error(error_msg)
                return False, None, error_msg

        except Exception as e:
            error_msg = f"Failed to resolve input {input_string}: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    def _analyze_input(self, input_string: str) -> Tuple[str, str]:
        """
        Analyze input string to determine type and extract relevant information

        Returns:
            Tuple[input_type: str, resolved_value: str]
        """
        input_clean = input_string.strip()

        # Check if it's a URL
        if input_clean.startswith(("http://", "https://")):
            parsed_url = urlparse(input_clean)

            # Check for Springer URLs
            if "springer.com" in parsed_url.netloc:
                # Extract DOI from Springer URL if possible
                doi_match = re.search(
                    r"/(?:book|article|chapter)/(?:10\.1007/)?([^/?]+)", input_clean
                )
                if doi_match:
                    doi = doi_match.group(1)
                    if not doi.startswith("10.1007/"):
                        doi = f"10.1007/{doi}"
                    return "springer_url", input_clean  # Return full URL, not just DOI
                else:
                    return "springer_url", input_clean

            # Check for DOI.org URLs
            elif "doi.org" in parsed_url.netloc:
                doi_part = input_clean.split("doi.org/")[-1]
                if self._is_springer_doi(doi_part):
                    return "springer_doi", doi_part
                else:
                    return "crossref_doi", doi_part

            # Generic URL
            else:
                return "generic_url", input_clean

        # Check if it's a DOI (starts with "10." and contains "/")
        elif input_clean.startswith("10.") and "/" in input_clean:
            if self._is_springer_doi(input_clean):
                return "springer_doi", input_clean
            else:
                return "crossref_doi", input_clean

        # Assume it's a DOI if nothing else matches
        else:
            if self._is_springer_doi(input_clean):
                return "springer_doi", input_clean
            else:
                return "crossref_doi", input_clean

    def _is_springer_doi(self, doi: str) -> bool:
        """Check if DOI is from Springer (starts with 10.1007) - Claude Generated"""
        return doi.startswith("10.1007")

    def _resolve_springer_url(
        self, url: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Resolve Springer URL using enhanced web crawling - Claude Generated"""
        try:
            # Create event loop for asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async crawling
            springer_data = loop.run_until_complete(self._crawl_springer_url(url))

            # Close event loop
            loop.close()

            if springer_data:
                # Extract abstract text for pipeline use
                abstract_text = self._extract_abstract_from_springer_data(springer_data)
                self.logger.info(
                    f"Successfully resolved Springer URL, abstract length: {len(abstract_text) if abstract_text else 0}"
                )
                return True, springer_data, abstract_text
            else:
                error_msg = f"No data received from Springer website for URL: {url}"
                self.logger.warning(error_msg)
                return False, None, error_msg

        except Exception as e:
            error_msg = f"Error parsing Springer website: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    def _resolve_springer_doi(
        self, doi: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Resolve Springer DOI using web crawling - Claude Generated"""
        try:
            # Create event loop for asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async crawling
            url = f"https://link.springer.com/book/{doi}"
            springer_data = loop.run_until_complete(self._crawl_springer_url(url))

            # Close event loop
            loop.close()

            if springer_data:
                # Extract abstract text for pipeline use
                abstract_text = self._extract_abstract_from_springer_data(springer_data)
                self.logger.info(
                    f"Successfully resolved Springer DOI, abstract length: {len(abstract_text) if abstract_text else 0}"
                )
                return True, springer_data, abstract_text
            else:
                error_msg = f"No data received from Springer website for DOI: {doi}"
                self.logger.warning(error_msg)
                return False, None, error_msg

        except Exception as e:
            error_msg = f"Error parsing Springer website: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    def _resolve_generic_url(
        self, url: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Resolve generic URL using web crawling - Claude Generated"""
        try:
            # Create event loop for asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run async crawling
            generic_data = loop.run_until_complete(self._crawl_generic_url(url))

            # Close event loop
            loop.close()

            if generic_data:
                # Extract comprehensive text content for pipeline use
                text_content = self._extract_content_from_generic_data(generic_data)
                self.logger.info(
                    f"Successfully crawled generic URL, content length: {len(text_content) if text_content else 0}"
                )
                return True, generic_data, text_content
            else:
                error_msg = f"No data received from URL: {url}"
                self.logger.warning(error_msg)
                return False, None, error_msg

        except Exception as e:
            error_msg = f"Error crawling generic URL: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    def _resolve_crossref_doi(
        self, doi: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Resolve DOI using CrossRef API - Claude Generated"""
        url = f"https://api.crossref.org/works/{doi}"

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                error_msg = f"API request failed: Status code {response.status_code}"
                self.logger.error(error_msg)
                return False, None, error_msg

            data = response.json()
            if data.get("status") != "ok":
                error_msg = f"API request not successful: {data.get('status')}"
                self.logger.error(error_msg)
                return False, None, error_msg

            message = data.get("message", {})

            # Extract relevant information
            result = {
                "Title": " | ".join(message.get("title", [])),
                "DOI": message.get("DOI", "Not available"),
                "Abstract": self._clean_jats(
                    message.get("abstract", "No abstract available")
                ),
                "Authors": self._format_authors(message.get("author", [])),
                "Publisher": message.get("publisher", "Not available"),
                "Published": self._format_date(
                    message.get("published-print", message.get("published-online", {}))
                ),
                "Type": message.get("type", "Not available"),
                "URL": message.get("URL", "Not available"),
            }

            # Extract abstract text for pipeline use
            abstract_text = result["Abstract"]
            self.logger.info(
                f"Successfully resolved CrossRef DOI, abstract length: {len(abstract_text) if abstract_text else 0}"
            )
            return True, result, abstract_text

        except Exception as e:
            error_msg = f"Error with CrossRef API: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    async def _crawl_springer_url(self, url: str):
        """Crawl Springer website for given URL - Claude Generated"""
        try:
            # Import here to avoid dependency issues if crawl4ai is not available
            from crawl4ai import AsyncWebCrawler

            # Create AsyncWebCrawler instance
            async with AsyncWebCrawler() as crawler:
                # Run crawler on URL
                result = await crawler.arun(url=url)

                # Parse markdown with enhanced table of contents extraction
                return self._parse_springer_markdown_enhanced(result.markdown, url)

        except ImportError:
            self.logger.error("crawl4ai not available for Springer URL resolution")
            raise Exception(
                "crawl4ai dependency not available for Springer URL resolution"
            )
        except Exception as e:
            raise Exception(f"Error crawling Springer website: {str(e)}")

    async def _crawl_generic_url(self, url: str):
        """Crawl generic website for given URL - Claude Generated"""
        try:
            # Import here to avoid dependency issues if crawl4ai is not available
            from crawl4ai import AsyncWebCrawler

            # Create AsyncWebCrawler instance
            async with AsyncWebCrawler() as crawler:
                # Run crawler on URL
                result = await crawler.arun(url=url)

                # Parse generic content
                return self._parse_generic_content(result.markdown, url)

        except ImportError:
            self.logger.error("crawl4ai not available for generic URL resolution")
            raise Exception(
                "crawl4ai dependency not available for generic URL resolution"
            )
        except Exception as e:
            raise Exception(f"Error crawling generic website: {str(e)}")

    def _parse_springer_markdown_enhanced(self, markdown_text: str, url: str) -> Dict:
        """Extract relevant information from Springer website markdown with enhanced table of contents - Claude Generated"""
        # Extract title (multiple patterns for better coverage)
        title_match = re.search(r"Book Title:\s+(.+?)(?:\n|$)", markdown_text)
        if not title_match:
            title_match = re.search(r"#\s+(.*?)$", markdown_text, re.MULTILINE)

        title = title_match.group(1).strip() if title_match else "Nicht verfügbar"

        # Extract "About this book" section
        about_match = re.search(
            r"## About this book\s+(.*?)(?=## Keywords|## Table of contents|$)",
            markdown_text,
            re.DOTALL,
        )
        about = about_match.group(1).strip() if about_match else "Nicht verfügbar"

        # Enhanced table of contents extraction - exact pattern from CrossrefWorker
        toc_match = re.search(
            r"## Table of contents.*?\n(.*?)(?=Back to top)", markdown_text, re.DOTALL
        )

        if toc_match:
            raw_toc = toc_match.group(1).strip()
            toc = self._clean_table_of_contents_enhanced(raw_toc)
        else:
            toc = "Nicht verfügbar"

        # Enhanced keywords extraction - exact pattern from CrossrefWorker
        keywords_match = re.search(
            r"## Keywords\s+(.*?)(?=Search within this book)", markdown_text, re.DOTALL
        )
        if keywords_match:
            raw_keywords = keywords_match.group(1).strip()
            keywords = self._clean_keywords_enhanced(raw_keywords)
        else:
            keywords = "Nicht verfügbar"

        # Extract authors/editors
        authors_match = re.search(
            r"Editors?:\s+(.*?)(?:\n\n)", markdown_text, re.DOTALL
        )
        authors = authors_match.group(1).strip() if authors_match else "Nicht verfügbar"
        authors = re.sub(r"\\$|\\$|\(|\)|https?://[^\s]+", "", authors).strip()

        # Extract publisher
        publisher_match = re.search(r"Publisher:\s+(.*?)(?:\n)", markdown_text)
        publisher = publisher_match.group(1).strip() if publisher_match else "Springer"

        # Extract publication date
        date_match = re.search(r"Published:\s+(\d{1,2}\s+\w+\s+\d{4})", markdown_text)
        published_date = (
            date_match.group(1).strip() if date_match else "Nicht verfügbar"
        )

        # Extract DOI if present in URL
        doi_match = re.search(r"/book/(?:10\.1007/)?([^/?]+)", url)
        doi = doi_match.group(1) if doi_match else "Nicht verfügbar"
        if doi != "Nicht verfügbar" and not doi.startswith("10.1007/"):
            doi = f"10.1007/{doi}"

        return {
            "Title": title,
            "DOI": doi,
            "Abstract": about,  # Use "About" section as abstract
            "Authors": authors,
            "Publisher": publisher,
            "Published": published_date,
            "Container-Title": "Springer Book",
            "URL": url,
            "About": about,
            "Table of Contents": toc,
            "Keywords": keywords,
            "Source": "Springer Enhanced",
        }

    def _parse_generic_content(self, markdown_text: str, url: str) -> Dict:
        """Parse generic website content - Claude Generated"""
        # Extract title (first heading)
        title_match = re.search(r"^#\s+(.*?)$", markdown_text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Not available"

        # Extract first paragraph as description
        # Split by double newlines and find first substantial paragraph
        paragraphs = markdown_text.split("\n\n")
        description = "Not available"
        for paragraph in paragraphs:
            clean_para = paragraph.strip()
            if len(clean_para) > 50 and not clean_para.startswith("#"):
                description = clean_para[:500] + (
                    "..." if len(clean_para) > 500 else ""
                )
                break

        return {
            "Title": title,
            "URL": url,
            "Content": markdown_text,
            "Description": description,
            "Source": "Generic Web Crawl",
        }

    def _extract_abstract_from_springer_data(self, springer_data: Dict) -> str:
        """Extract comprehensive abstract text from Springer data including ToC and keywords - Claude Generated"""
        abstract_parts = []

        # Add title
        title = springer_data.get("Title", "")
        if title and title not in ["Not available", "Nicht verfügbar"]:
            abstract_parts.append(f"Titel: {title}")

        # Add "About" section as main abstract
        about = springer_data.get("About", "")
        if about and about not in ["Not available", "Nicht verfügbar"]:
            abstract_parts.append(f"Zusammenfassung:\n{about}")

        # Add table of contents - this was missing!
        toc = springer_data.get("Table of Contents", "")
        if toc and toc not in ["Not available", "Nicht verfügbar"]:
            abstract_parts.append(f"Inhaltsverzeichnis:\n{toc}")

        # Add keywords - this was missing!
        keywords = springer_data.get("Keywords", "")
        if keywords and keywords not in ["Not available", "Nicht verfügbar"]:
            abstract_parts.append(f"Schlüsselwörter: {keywords}")

        # Add authors if available
        authors = springer_data.get("Authors", "")
        if authors and authors not in ["Not available", "Nicht verfügbar"]:
            abstract_parts.append(f"Autoren/Herausgeber: {authors}")

        if abstract_parts:
            return "\n\n".join(abstract_parts)

        return "Keine Daten von Springer verfügbar"

    def _extract_content_from_generic_data(self, generic_data: Dict) -> str:
        """Extract comprehensive content from generic web crawl data - Claude Generated"""
        content_parts = []

        # Add title
        title = generic_data.get("Title", "")
        if title and title != "Not available":
            content_parts.append(f"Titel: {title}")

        # Add description if available
        description = generic_data.get("Description", "")
        if description and description != "Not available":
            content_parts.append(f"Beschreibung:\n{description}")

        # Add full content (markdown)
        full_content = generic_data.get("Content", "")
        if full_content:
            # Clean markdown for better readability
            cleaned_content = self._clean_markdown_content(full_content)
            if len(cleaned_content) > 1000:
                content_parts.append(f"Inhalt:\n{cleaned_content[:2000]}...")
            else:
                content_parts.append(f"Inhalt:\n{cleaned_content}")

        if content_parts:
            return "\n\n".join(content_parts)

        return "Keine Inhalte von URL verfügbar"

    def _clean_markdown_content(self, content: str) -> str:
        """Clean markdown content for better pipeline processing - Claude Generated"""
        if not content:
            return ""

        # Remove markdown formatting
        # Remove headers but keep the text
        content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
        # Remove markdown links but keep the text
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
        # Remove emphasis but keep the text
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
        content = re.sub(r"\*([^*]+)\*", r"\1", content)
        # Clean up extra whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)
        content = re.sub(r"[ \t]+", " ", content)

        return content.strip()

    def _clean_table_of_contents_enhanced(self, raw_toc: str) -> str:
        """Enhanced table of contents cleaning - exact copy from working CrossrefWorker - Claude Generated"""
        if not raw_toc:
            return "Nicht verfügbar"

        # Find all texts in square brackets, excluding download links
        # Pattern: [Text] but not [Download chapter PDF]
        chapter_pattern = r"\[([^\]]+)\]"

        chapters = []
        matches = re.findall(chapter_pattern, raw_toc)

        for match in matches:
            # Filter out download links
            if not any(
                keyword in match.lower()
                for keyword in ["download", "pdf", "chapter pdf", "Back to top"]
            ):
                # Clean the title
                clean_title = match.strip()
                if clean_title and clean_title not in chapters:
                    chapters.append(clean_title)

        # Format table of contents as numbered list
        if chapters:
            formatted_toc = "\n".join(
                [f"{i+1}. {chapter}" for i, chapter in enumerate(chapters)]
            )
            return formatted_toc
        else:
            return "Nicht verfügbar"

    def _clean_keywords_enhanced(self, raw_keywords: str) -> str:
        """Enhanced keywords cleaning - exact copy from working CrossrefWorker - Claude Generated"""
        if not raw_keywords:
            return "Nicht verfügbar"

        # Find all texts in square brackets (keywords)
        keyword_pattern = r"\[([^\]]+)\]"

        keywords = []
        matches = re.findall(keyword_pattern, raw_keywords)

        for match in matches:
            # Clean the keyword
            clean_keyword = match.strip()
            if clean_keyword and clean_keyword not in keywords:
                keywords.append(clean_keyword)

        # Format keywords as comma-separated list
        if keywords:
            return ", ".join(keywords)
        else:
            return "Nicht verfügbar"

    def _clean_table_of_contents(self, raw_toc: str) -> str:
        """Clean and format table of contents - Claude Generated"""
        lines = raw_toc.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("*") and not line.startswith("-"):
                # Remove markdown links but keep the text
                line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines) if cleaned_lines else "Not available"

    def _clean_jats(self, text: str) -> str:
        """Clean JATS XML formatting from text - Claude Generated"""
        if not text or text == "No abstract available":
            return text

        # Remove JATS XML tags
        text = re.sub(r"<[^>]+>", "", str(text))
        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _format_authors(self, authors_list: list) -> str:
        """Format authors list into readable string - Claude Generated"""
        if not authors_list:
            return "Not available"

        formatted_authors = []
        for author in authors_list:
            given = author.get("given", "")
            family = author.get("family", "")
            if given and family:
                formatted_authors.append(f"{given} {family}")
            elif family:
                formatted_authors.append(family)

        return ", ".join(formatted_authors) if formatted_authors else "Not available"

    def _format_date(self, date_dict: dict) -> str:
        """Format date dictionary into readable string - Claude Generated"""
        if not date_dict:
            return "Not available"

        parts = date_dict.get("date-parts", [[]])
        if parts and parts[0]:
            date_parts = parts[0]
            if len(date_parts) >= 3:
                return f"{date_parts[2]:02d}.{date_parts[1]:02d}.{date_parts[0]}"
            elif len(date_parts) >= 2:
                return f"{date_parts[1]:02d}.{date_parts[0]}"
            elif len(date_parts) >= 1:
                return str(date_parts[0])

        return "Not available"


# Backward compatibility class
class DOIResolver(UnifiedResolver):
    """Backward compatibility class for existing DOI resolution code - Claude Generated"""

    def resolve_doi(self, doi: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Resolve DOI using unified resolver for backward compatibility - Claude Generated"""
        return self.resolve(doi)


# Convenience functions for simple usage
def resolve_doi_to_text(
    doi: str, logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Simple function to resolve DOI to abstract text (backward compatibility)

    Args:
        doi: DOI string
        logger: Optional logger

    Returns:
        Tuple[success: bool, abstract_text: str, error_message: str]
    """
    resolver = UnifiedResolver(logger)
    success, metadata, result = resolver.resolve(doi)

    if success:
        return True, result, None
    else:
        return False, None, result


def resolve_url_to_text(
    url: str, logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Simple function to resolve URL to text content

    Args:
        url: URL string
        logger: Optional logger

    Returns:
        Tuple[success: bool, text_content: str, error_message: str]
    """
    resolver = UnifiedResolver(logger)
    success, metadata, result = resolver.resolve(url)

    if success:
        return True, result, None
    else:
        return False, None, result


def resolve_input_to_text(
    input_string: str, logger: Optional[logging.Logger] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Universal function to resolve DOI, URL, or DOI URL to text content

    Args:
        input_string: DOI, URL, or DOI URL
        logger: Optional logger

    Returns:
        Tuple[success: bool, text_content: str, error_message: str]
    """
    resolver = UnifiedResolver(logger)
    success, metadata, result = resolver.resolve(input_string)

    if success:
        return True, result, None
    else:
        return False, None, result
