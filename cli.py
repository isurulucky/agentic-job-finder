import argparse
import asyncio
from datetime import date
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

from browser_nav_agent import BrowserNavigationAgent
from nav_experience_store import NavExperienceStore
from vacancy_filter_agent import VacancyFilterAgent
from url_finder import find_candidate_urls, identify_framework
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
        help="LLM model (default: gpt-5.4-mini).",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider (default: openai).",
    )
    parser.add_argument(
        "--headless",
        default=False,
        type=bool,
        help="Run Playwright in headless mode/not.",
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

    llm = ChatOpenAI(
        model=args.model,
        reasoning={"effort": "medium"},
        use_responses_api=True,
    )

    experience_store = NavExperienceStore()

    # Retrieve previous navigation experiences for each website.
    experiences: dict[str, list[dict]] = {}

    for target_url in target_urls:
        framework = identify_framework(target_url)
        previous_records = experience_store.retrieve(
            framework=framework,
            limit=10,
        )

        experiences[target_url] = [
            record["experiences"]
            for record in previous_records
        ]

        print(
            f"[*] Retrieved {len(experiences[target_url])} "
            f"experiences for {target_url} ({framework})."
        )

    # Discover candidate vacancy URLs using previous experiences.
    browser_agent = BrowserNavigationAgent(
        keywords=args.keywords,
        location=args.location,
        language=args.language,
        llm=llm,
    )

    navigation_results = await browser_agent.run(
        target_urls=target_urls,
        experiences=experiences,
        headless=args.headless,
    )

    # Store successful navigation experiences and collect vacancy links.
    links: list[JobLink] = []

    for result in navigation_results:
        framework = identify_framework(result.url)

        if result.result_count > 0:
            experience_id = experience_store.store(
                framework=framework,
                result_count=result.result_count,
                experiences=result.experience,
            )

            if experience_id:
                print(
                    f"[*] Stored navigation experience for "
                    f"{result.url} ({framework})."
                )
        else:
            print(
                f"[*] No vacancy links found for {result.url}; "
                "experience not stored."
            )

        links.extend(result.job_links)

    # Filter candidates deterministically using URL patterns and ATS domains.
    candidate_links = find_candidate_urls(
        [
            {"url": link.url, "text": link.text}
            for link in links
        ]
    )

    filtered_links = [
        JobLink(url=link["url"], text=link["text"])
        for link in candidate_links
    ]

    print(
        f"[*] {len(links)} links discovered; "
        f"{len(filtered_links)} candidates remain after URL filtering."
    )

    # Identify semantically matching vacancies.
    filtering_agent = VacancyFilterAgent(
        llm=llm,
        keywords=args.keywords,
        location=args.location,
        language=args.language,
    )

    job_links = await filtering_agent.run(filtered_links)

    print(f"\n[*] Found {len(job_links)} matching vacancies:")

    for job_link in job_links:
        print(job_link.url)


if __name__ == "__main__":
    asyncio.run(main())

