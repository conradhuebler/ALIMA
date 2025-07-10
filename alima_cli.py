
import argparse
import logging
import os
from src.core.alima_manager import AlimaManager
from src.llm.llm_service import LlmService
from src.llm.prompt_service import PromptService
from src.core.data_models import AbstractData

PROMPTS_FILE = "prompts.json"

def main():
    """Main function for the ALIMA CLI."""
    parser = argparse.ArgumentParser(description="ALIMA CLI - AI-powered abstract analysis.")
    parser.add_argument("task", help="The analysis task to perform (e.g., 'abstract', 'keywords').")
    parser.add_argument("--abstract", required=True, help="The abstract or text to analyze.")
    parser.add_argument("--keywords", help="Optional keywords to include in the analysis.")
    parser.add_argument("--model", required=True, help="The model to use for the analysis.")
    parser.add_argument("--provider", default="ollama", help="The LLM provider to use (e.g., 'ollama', 'gemini').")
    parser.add_argument("--ollama-host", default="http://localhost", help="Ollama host URL.")
    parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama port.")
    parser.add_argument("--use-chunking-abstract", action="store_true", help="Enable chunking for the abstract.")
    parser.add_argument("--abstract-chunk-size", type=int, default=100, help="Chunk size for the abstract.")
    parser.add_argument("--use-chunking-keywords", action="store_true", help="Enable chunking for keywords.")
    parser.add_argument("--keyword-chunk-size", type=int, default=500, help="Chunk size for keywords.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Check if prompts file exists
    if not os.path.exists(PROMPTS_FILE):
        logger.error(f"Prompts file not found at: {PROMPTS_FILE}")
        return

    # Setup services
    llm_service = LlmService(ollama_url=args.ollama_host, ollama_port=args.ollama_port)
    prompt_service = PromptService(PROMPTS_FILE, logger)
    alima_manager = AlimaManager(llm_service, prompt_service, logger)

    # Prepare data
    abstract_data = AbstractData(abstract=args.abstract, keywords=args.keywords)

    # Run analysis
    try:
        result = alima_manager.analyze_abstract(
            abstract_data,
            args.task,
            args.model,
            args.use_chunking_abstract,
            args.abstract_chunk_size,
            args.use_chunking_keywords,
            args.keyword_chunk_size,
            provider=args.provider
        )
        print("--- Analysis Result ---")
        print(result.full_text)
        print("--- Matched Keywords ---")
        print(result.matched_keywords)
        print("--- GND Systematic ---")
        print(result.gnd_systematic)

    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
