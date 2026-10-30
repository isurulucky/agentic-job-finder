from langchain_core.language_models import BaseChatModel

from tools import JobLink, MatchingVacancies


class VacancyFilterAgent:
    """Identifies vacancies that semantically match the search criteria."""

    def __init__(self, llm: BaseChatModel, keywords: str, location: str, language: str) -> None:
        self.llm = llm
        self.keywords = keywords
        self.location = location
        self.language = language

        self.extractor = self.llm.with_structured_output(
            MatchingVacancies,
            method="json_schema",
        )

    async def run(self, job_links: list[JobLink]) -> list[JobLink]:
        """Return all candidate vacancies matching the search criteria."""

        if not job_links:
            print("Zero candidate URLs, no matching vacancies.")
            return []

        extract_prompt = f"""
        Search criteria: {self.keywords}
        Preferred location: {self.location}
        Preffered language: {self.language}

        Candidate vacancies:
        {[
            {"url": job.url, "text": job.text}
            for job in job_links
        ]}

        Select every vacancy that matches the search criteria, preferred location and language.

        Consider both direct keyword matches and contextual similarity.
        Do not return only the best match.
        Do not invent URLs or return vacancies that are not in the candidates.
        Return all matching vacancies in the job_links field.
        """

        result = await self.extractor.ainvoke(extract_prompt)

        return result.job_links
