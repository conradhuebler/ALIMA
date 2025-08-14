import asyncio
import re
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crawl4ai import AsyncWebCrawler

async def main():
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on a URL
        for i in range(1, 36):
            result = await crawler.arun(url=f"https://katalog.ub.tu-freiberg.de/Record/0-1856236676?sid=51925798")
            print(result.markdown)
            return

# Run the async main function
asyncio.run(main())
