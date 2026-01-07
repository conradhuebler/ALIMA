from typing import List, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)


def chunk_abstract_by_lines(text: str, lines_per_chunk: int) -> List[str]:
    logger.info(
        f"chunk_abstract_by_lines called with text length {len(text)} and {lines_per_chunk} lines per chunk"
    )
    lines = text.split("\n")
    chunks = []

    for i in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[i : i + lines_per_chunk]
        chunks.append("\n".join(chunk_lines))

    logger.info(
        f"Split abstract into {len(chunks)} chunks of {lines_per_chunk} lines each"
    )
    return chunks


def chunk_keywords_by_comma(keywords_text: str, keywords_per_chunk: int) -> List[str]:
    logger.info(
        f"chunk_keywords_by_comma called with keywords_text length {len(keywords_text)} and {keywords_per_chunk} keywords per chunk"
    )
    keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
    chunks = []

    for i in range(0, len(keywords), keywords_per_chunk):
        chunk_keywords = keywords[i : i + keywords_per_chunk]
        chunks.append(", ".join(chunk_keywords))

    logger.info(
        f"Split keywords into {len(chunks)} chunks of {keywords_per_chunk} keywords each"
    )
    return chunks


def parse_keywords_from_list(keywords_string: str) -> Dict[str, str]:
    # logger.info(
    #    f"parse_keywords_from_list called with keywords_string: '{keywords_string}'"
    # )
    keywords_dict = {}
    # logger.info(f"Parsing keywords from string: '{keywords_string}'")

    if not keywords_string.strip():
        logger.debug("Keywords string is empty or whitespace only.")
        return keywords_dict

    # Split by comma and process each entry
    entries = [entry.strip() for entry in keywords_string.split(",")]
    # logger.info(f"Split into entries: {entries}")

    for entry in entries:
        keyword = entry
        gnd_match = None
        if "(" in entry and ")" in entry:
            # Extract keyword and GND number
            try:
                keyword = entry.split("(")[0].strip()
                gnd_match = entry.split("(")[1].split(")")[0].strip()
            except IndexError:
                logger.warning(f"Failed to parse entry (IndexError): '{entry}'")

        if keyword:
            keywords_dict[keyword] = gnd_match
            logger.debug(f"Parsed: Keyword='{keyword}', GND='{gnd_match}'")
        else:
            logger.debug(f"Skipped entry (empty keyword): '{entry}'")

    logger.debug(f"Final parsed keywords_dict: {keywords_dict}")
    return keywords_dict


def extract_keywords_from_response(text: str) -> str:
    logger.info(f"extract_keywords_from_response called with text length {len(text)}")
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"<final_list>(.*?)</final_list>", cleaned_text, re.DOTALL)
    if match:
        keywords = match.group(1).split("|")
        result = ", ".join([keyword.strip().replace("_", " ") for keyword in keywords if keyword.strip()])
        logger.debug(f"Extracted keywords: {result}")
        return result
    logger.debug("No <final_list> tag found.")
    return ""


def extract_title_from_response(text: str) -> Optional[str]:
    """
    Extract work title from LLM response <final_title> tags - Claude Generated

    Args:
        text: LLM response text

    Returns:
        Extracted title string or None if not found
    """
    logger.info(f"extract_title_from_response called with text length {len(text)}")

    # Remove <think> tags first (same pattern as extract_keywords_from_response)
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Extract <final_title> content
    match = re.search(r"<final_title>(.*?)</final_title>", cleaned_text, re.DOTALL)
    if match:
        # Extract and clean the title
        title = match.group(1).strip()

        # Remove newlines and excessive whitespace
        title = ' '.join(title.split())

        logger.debug(f"Extracted title: '{title}'")
        return title

    logger.debug("No <final_title> tag found.")
    return None


def extract_gnd_system_from_response(text: str) -> Optional[str]:
    logger.info(f"extract_gnd_system_from_response called with text length {len(text)}")
    try:
        # Remove <think> tags first
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Try to find the <class> tag first (if the prompt is updated to include it)
        match_class_tag = re.search(r"<class>(.*?)</class>", cleaned_text, re.DOTALL)
        if match_class_tag:
            ognd_system = match_class_tag.group(1).strip()
            logger.debug(f"Extracted GND system from <class> tag: {ognd_system}")
            return ognd_system

        # If <class> tag not found, try to extract from 'GND-Systematik:' section
        match_gnd_section = re.search(
            r"GND-Systematik:\s*(.*?)(?:\n\nSchlagworte:|\n\nZerlegte Schlagworte:|\n\nFEHLENDE KONZEPTE:|\n\nKONKRETE FEHLENDE OBERBEGRIFFE BZW. SCHLAGWORTE:|<final_list>|\Z)",
            cleaned_text,
            re.DOTALL,
        )
        if match_gnd_section:
            ognd_system = match_gnd_section.group(1).strip()
            # Remove any lines that are section headers
            lines = ognd_system.splitlines()
            filtered_lines = []
            for line in lines:
                if not re.match(
                    r"^(Schlagworte:|Zerlegte Schlagworte:|FEHLENDE KONZEPTE:|KONKRETE FEHLENDE OBERBEGRIFFE BZW. SCHLAGWORTE:)",
                    line.strip(),
                ):
                    filtered_lines.append(line.strip())
            ognd_system = "\n".join([line for line in filtered_lines if line])
            logger.debug(
                f"Extracted GND system from 'GND-Systematik:' section: {ognd_system}"
            )
            return ognd_system

        logger.debug("No <class> tag or 'GND-Systematik:' section found.")
        return None
    except Exception as e:
        logger.error(f"Error extracting GND system: {str(e)}")
        return None


def match_keywords_against_text(
    keywords_dict: Dict[str, str], text: str
) -> Dict[str, str]:
    logger.info(
        f"match_keywords_against_text called with {len(keywords_dict)} keywords and text length {len(text)}"
    )
    matched_keywords = {}
    for keyword, gnd_id in keywords_dict.items():
        # Create a regex for whole word, case-insensitive match
        exact_match_pattern = r"\\b" + re.escape(keyword) + r"\\b"
        if re.search(exact_match_pattern, text, re.IGNORECASE):
            matched_keywords[keyword] = gnd_id
            logger.debug(f"Exact match found: '{keyword}' -> {gnd_id}")
    logger.debug(f"Final matched keywords: {matched_keywords}")
    return matched_keywords
