from urllib.parse import urlparse

JOB_URL_PATTERNS = [
    "/jobs",
    "/job/",
    "/careers",
    "/career/",
    "/vacancies",
    "/vacancy/",
    "/openings",
    "/positions",
    "/opportunities",
    "/employment",
    "/recruitment",
    "/job-search",
    "/job-openings",
    "/current-openings",
]

ATS_PLATFORMS = [
    "greenhouse.io",
    "lever.co",
    "myworkdayjobs.com",
    "ashbyhq.com",
    "smartrecruiters.com",
    "jobvite.com",
    "workable.com",
    "icims.com",
    "bamboohr.com",
    "recruitee.com",
    "teamtailor.com",
    "personio.com",
    "successfactors.com",
    "taleo.net",
]

def filter_job_links(
    links: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Filters links using job-related URL patterns and ATS domains."""

    filtered_links = []

    for link in links:
        url = link.get("url", "").strip()

        if not url:
            continue

        parsed_url = urlparse(url)
        hostname = (parsed_url.hostname or "").lower()
        path = parsed_url.path.lower()

        is_job_url = any(
            pattern in path
            for pattern in JOB_URL_PATTERNS
        )

        is_ats_url = any(
            hostname == platform or hostname.endswith("." + platform)
            for platform in ATS_PLATFORMS
        )

        if is_job_url or is_ats_url:
            filtered_links.append(link)

    return filtered_links