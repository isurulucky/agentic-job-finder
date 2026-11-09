from pydantic import BaseModel, Field
from tools import JobLink

class WebsiteNavigationResult(BaseModel):
    url: str
    job_links: list[JobLink] = Field(default_factory=list)
    experience: dict = Field(default_factory=dict)
    result_count: int = 0
    termination_reason: str = "unknown"

def normalize_job_links(result: object) -> list[JobLink]:
    """Validate and normalize a list of job vacancy links."""
    if not isinstance(result, list):
        raise TypeError(
            f"Expected a list of JobLink objects, got {type(result).__name__}"
        )

    return [
        item if isinstance(item, JobLink) else JobLink.model_validate(item)
        for item in result
    ]