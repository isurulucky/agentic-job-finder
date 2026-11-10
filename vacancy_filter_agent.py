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
        You are a vacancy matching assistant.

        Search criteria:
        - Job keywords: {self.keywords}
        - Preferred location: {self.location}
        - Preferred language: {self.language}

        Candidate vacancies:
        {[
            {"url": job.url, "text": job.text}
            for job in job_links
        ]}

        Your task is to identify ALL potentially relevant vacancies.

        Matching instructions:

        1. Job relevance:
        - Consider job titles, responsibilities, technologies, and required skills.
        - Consider direct keyword matches as well as semantic similarity.
        - Include related engineering roles even if their titles do not exactly match the search keywords.
        - Do not require an exact keyword match.

        2. Location:
        - Prefer vacancies in the requested location.
        - Do not automatically reject a vacancy if its location is missing.
        - Consider relocation possibilities where relevant.

        3. Language:
        - Consider the preferred language when evaluating vacancies.
        - Do not reject a vacancy merely because its language requirements are not mentioned in the available text.

        4. Incomplete information:
        - A short title or incomplete description is not sufficient evidence to reject an otherwise relevant vacancy.
        - If a vacancy could reasonably match the criteria, include it.

        5. Output:
        - Favor recall over precision.
        - Return every potentially relevant vacancy, not just the best matches.
        - Never invent URLs.
        - Only return URLs from the candidate list.
        - Return matching vacancies in the job_links field.
        """

        result = await self.extractor.ainvoke(extract_prompt)

        return result.job_links
