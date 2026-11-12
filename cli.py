import argparse
import asyncio
from datetime import date
from langchain.chat_models import init_chat_model

from browser_nav_agent import BrowserNavigationAgent
from experience_based_nav import ExperienceBasedNavigator
from model_factory import create_llm
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

    llm = create_llm(args.provider, args.model)

    experience_store = NavExperienceStore()

    # Retrieve previous navigation experiences for each website.
    experiences: dict[str, list[dict]] = {}
    target_urls_with_exp = []
    target_urls_without_exp = []

    for target_url in target_urls:
        framework = identify_framework(target_url)

        previous_records = experience_store.retrieve(
            framework=framework,
            limit=10,
        )

        if previous_records:
            experiences[target_url] = [
                record["experiences"]
                for record in previous_records
            ]
            target_urls_with_exp.append(target_url)
        else:
            target_urls_without_exp.append(target_url)

        print(
            f"\n[*] Retrieved {len(experiences.get(target_url, []))} "
            f"experiences for {target_url} ({framework})."
        )


    # Attempt deterministic navigation for websites with experiences.
    exp_based_navigator = ExperienceBasedNavigator(
        keywords=args.keywords,
        location=args.location,
        language=args.language,
    )

    nav_results = await exp_based_navigator.run(
        target_urls=target_urls_with_exp,
        experiences=experiences,
        headless=args.headless,
    )


    # Identify websites requiring LLM navigation.
    failed_urls = [
        result.url
        for result in nav_results
        if not (
            result.termination_reason == "extraction_complete"
            and result.result_count > 0
        )
    ]


    if failed_urls:
        deleted_count = experience_store.delete_by_target_urls(
            failed_urls
        )

        print(
            f"\n[*] Deleted {deleted_count} failed navigation experiences "
            f"for {len(failed_urls)} URLs."
        )

    # Include websites without previous experiences.
    urls_for_agent = target_urls_without_exp + failed_urls


    # Use the LLM navigator for new websites and failed replays.
    browser_nav_agent = BrowserNavigationAgent(
        keywords=args.keywords,
        location=args.location,
        language=args.language,
        llm=llm,
    )

    agentic_nav_results = await browser_nav_agent.run(
        target_urls=urls_for_agent,
        headless=args.headless,
    )

    # Combine successful navigation results.
    successful_nav_results = [
        result
        for result in nav_results
        if result.termination_reason == "extraction_complete"
        and result.result_count > 0
    ]

    successful_agentic_results = [
        result
        for result in agentic_nav_results
        if result.termination_reason == "extraction_complete"
        and result.result_count > 0
    ]

    # Store successful experiences discovered by navigation agent, and collect vacancy links.
    links: list[JobLink] = []

    for result in successful_agentic_results:
        framework = identify_framework(result.url)

        if result.experience:
            experience_id = experience_store.store(
                framework=framework,
                result_count=result.result_count,
                experiences=result.experience,
            )

            if experience_id:
                print(
                    f"\n[*] Stored navigation experience for "
                    f"{result.url} ({framework})."
                )

        links.extend(result.job_links)

    for result in successful_nav_results:
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
        f"\n[*] {len(links)} links discovered; "
        f"{len(filtered_links)} candidates remain after URL filtering."
    )

    # for link in filtered_links: 
    #     print(f"{link.url}\n{link.text}\n------------------")
        
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

