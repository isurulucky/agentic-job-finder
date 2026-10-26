import argparse
import asyncio
from datetime import date
from langchain.chat_models import init_chat_model

from browser_nav_agent import BrowserNavigationAgent
from vacancy_filter_agent import VacancyFilterAgent
from url_finder import find_candidate_urls
from tools import JobLink

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find matching job vacancies across careers websites."
    )

    parser.add_argument(
        "--urls",
        required=True,
        help="Comma-separated vacancy site URLs.",
    )
    parser.add_argument(
        "--keywords",
        required=True,
        help="Job search keywords.",
    )
    parser.add_argument(
        "--location",
        required=True,
        help="Target job location.",
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Required vacancy language.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="LLM model (default: o4-mini).",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider (default: openai).",
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    target_urls = [
        url.strip()
        for url in args.urls.split(",")
        if url.strip()
    ]

    if not target_urls:
        raise SystemExit("Error: provide at least one vacancy site URL.")

    llm = init_chat_model(
        model=args.model,
        model_provider=args.provider,
    )

    # Discover candidate vacancy URLs.
    browser_agent = BrowserNavigationAgent(
        keywords=args.keywords,
        location=args.location,
        language=args.language,
        llm=llm,
    )
    links = await browser_agent.run(
        target_urls=target_urls,
        headless=False,
    )

    # Filter candidates deterministically.
    # TODO: remove convert to dictionaries 
    links = [
        {"url": cl.url, "text": cl.text}
        for cl in links
    ]

    # Filter using URL patterns and ATS domains
    candidate_links = find_candidate_urls(links)

    filtered_links = [
        JobLink(url=link["url"], text=link["text"]) for link in candidate_links
    ]

    print(f"No. of links after filtering: {len(filtered_links)}")

    # Identify semantically matching vacancies.
    filtering_agent = VacancyFilterAgent(
        llm=llm,
        keywords=args.keywords,
    )
    
    job_links = await filtering_agent.run(filtered_links)

    for jl in job_links:
        print(f"{jl.url}")


if __name__ == "__main__":
    asyncio.run(main())

