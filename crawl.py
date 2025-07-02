import asyncio
import re
from crawl4ai import AsyncWebCrawler

async def main():
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on a URL
        for i in range(1, 36):
            result = await crawler.arun(url=f"https://link.springer.com/book/10.1007/1-4020-2151-8")
            print(result.markdown)
            return
            pattern = r"## hat Element(.*?)\n\n"
            match = re.search(pattern, result.markdown, re.DOTALL)

            if match:
                extracted_data = match.group(1)
                print(extracted_data)
                link_pattern = r"https://sta\.dnb\.de/doc/GND-SY*"
                links = re.findall(link_pattern, extracted_data)

                print("\nExtracted links:")
                for link in links:
                    print(f"{link[0]}: {link[1]}")
            else:
    #            print("No match found.")
                print(result.markdown)


# Run the async main function
asyncio.run(main())
