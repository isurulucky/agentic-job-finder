import argparse
import asyncio
from datetime import date

from job_finder import AgentLoop

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
        default="o4-mini",
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

    agent_loop = AgentLoop(
        target_urls=target_urls,
        keywords=args.keywords,
        location=args.location,
        language=args.language,
    )

    openings = await agent_loop.run(
        model=args.model,
        model_provider=args.provider,
    )

    today = date.today().strftime("%d/%m/%Y")
    print(f"\nFound {len(openings)} matching openings on {today}.")

    for url, title in openings.items():
        print(f"{url}")


if __name__ == "__main__":
    asyncio.run(main())

