from typing import List, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)

def chunk_abstract_by_lines(text: str, lines_per_chunk: int) -> List[str]:
    logger.info(f"chunk_abstract_by_lines called with text length {len(text)} and {lines_per_chunk} lines per chunk")
    lines = text.split("\n")
    chunks = []

    for i in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[i : i + lines_per_chunk]
        chunks.append("\n".join(chunk_lines))

    logger.info(
        f"Split abstract into {len(chunks)} chunks of {lines_per_chunk} lines each"
    )
    return chunks

def chunk_keywords_by_comma(
    keywords_text: str, keywords_per_chunk: int
) -> List[str]:
    logger.info(f"chunk_keywords_by_comma called with keywords_text length {len(keywords_text)} and {keywords_per_chunk} keywords per chunk")
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
    logger.info(f"parse_keywords_from_list called with keywords_string: '{keywords_string}'")
    keywords_dict = {}
    logger.info(f"Parsing keywords from string: '{keywords_string}'")

    if not keywords_string.strip():
        logger.debug("Keywords string is empty or whitespace only.")
        return keywords_dict

    # Split by comma and process each entry
    entries = [entry.strip() for entry in keywords_string.split(",")]
    logger.info(f"Split into entries: {entries}")

    for entry in entries:
        if "(" in entry and ")" in entry:
            # Extract keyword and GND number
            try:
                keyword = entry.split("(")[0].strip()
                gnd_match = entry.split("(")[1].split(")")[0].strip()

                if keyword and gnd_match:
                    keywords_dict[keyword] = gnd_match
                    logger.debug(f"Parsed: Keyword='{keyword}', GND='{gnd_match}'")
                else:
                    logger.debug(f"Skipped entry (empty keyword or GND): '{entry}'")
            except IndexError:
                logger.warning(f"Failed to parse entry (IndexError): '{entry}'")
        else:
            logger.debug(f"Skipped entry (no parentheses): '{entry}'")

    logger.debug(f"Final parsed keywords_dict: {keywords_dict}")
    return keywords_dict

def extract_keywords_from_response(text: str) -> str:
    logger.info(f"extract_keywords_from_response called with text length {len(text)}")
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"<final_list>(.*?)</final_list>", cleaned_text, re.DOTALL)
    if match:
        keywords = match.group(1).split("|")
        quoted_keywords = [
            f'"{keyword.strip()}"' for keyword in keywords if keyword.strip()
        ]
        result = ", ".join(quoted_keywords)
        logger.debug(f"Extracted keywords: {result}")
        return result
    logger.debug("No <final_list> tag found.")
    return ""

def extract_gnd_system_from_response(text: str) -> Optional[str]:
    logger.info(f"extract_gnd_system_from_response called with text length {len(text)}")
    try:
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        match = re.search(r"<class>(.*?)</class>", cleaned_text, re.DOTALL)
        if match:
            ognd_system = match.group(1).strip()
            logger.debug(f"Extracted GND system: {ognd_system}")
            return ognd_system
        logger.debug("No <class> tag found.")
        return None
    except Exception as e:
        logger.error(f"Error extracting class content: {str(e)}")
        return None

def match_keywords_against_text(keywords_dict: Dict[str, str], text: str) -> Dict[str, str]:
    logger.info(f"match_keywords_against_text called with {len(keywords_dict)} keywords and text length {len(text)}")
    matched_keywords = {}
    for keyword, gnd_id in keywords_dict.items():
        # Create a regex for whole word, case-insensitive match
        exact_match_pattern = r"\\b" + re.escape(keyword) + r"\\b"
        if re.search(exact_match_pattern, text, re.IGNORECASE):
            matched_keywords[keyword] = gnd_id
            logger.debug(f"Exact match found: '{keyword}' -> {gnd_id}")
    logger.debug(f"Final matched keywords: {matched_keywords}")
    return matched_keywords

