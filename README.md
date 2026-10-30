# Agentic JobFinder

Agentic JobFinder is a command-line application that discovers job vacancies matching specified keywords, location, and language across multiple careers websites.

The application uses a **multi-agent architecture** with specialized agents for navigating careers websites, extracting vacancy URLs and identifying vacancies that semantically match the search criteria. Deterministic filtering complements the agents to improve efficiency and reduce irrelevant results.

**Experience-based self-learning** enables the application to improve its navigation of careers websites over time. It stores successful navigation experiences associated with applicant tracking system (ATS) platforms and other career site providers, then retrieves and reuses relevant experiences during subsequent searches. This allows the navigation agent to leverage previously successful strategies when visiting familiar platforms while adapting its approach to unfamiliar websites.

Navigation experiences are currently stored in a local SQL database using SQLite. This lightweight persistence mechanism allows the application to retain and reuse experiences across separate runs.

Playwright powers browser automation, while LangChain provides the agent orchestration and LLM integration. The current implementation uses **OpenAI models through LangChain's `ChatOpenAI` integration and the OpenAI Responses API**, and depends on `OPENAI_API_KEY` environment variable.

Matching vacancy URLs are printed to standard output.

### How to Use

- Install the dependencies using `pip install -r requirements.txt`.
- Navigate to the project directory.
- Generate your API key and set it as a environment variable; for Open AI models, set the `OPENAI_API_KEY` environment variable.
- Run the application using `python cli.py`.

The following command-line arguments are supported:

- Mandatory `--urls`: Comma-separated URLs of careers or vacancy websites to search.
- Mandatory `--keywords`: Job search keywords, such as `Software Engineer` or `Python Developer`.
- Mandatory `--location`: Target job location, such as `Europe` or `Netherlands`.
- Mandatory `--language`: Required vacancy language, such as `English`.
- Optional `--headless`: Runs the browser in headless mode if specified as `True`. 

### Example

```bash
python cli.py \
  --urls "https://www.coolblue.nl/en/vacancies,https://catawiki.careers/vacancies,https://careers.adyen.com/" \
  --keywords "Software Engineer" \
  --location "Europe" \
  --language "English"
```

<!-- Use a different model:

```bash 
python cli.py \
  --urls "https://careers.example.com" \
  --keywords "Python Developer" \
  --location "Netherlands" \
  --language "English" \
  --model "claude-sonnet-4-6" \
  --provider "anthropic"
``` -->

## Running as a Cron Job

The application can be scheduled as a cron job to periodically search careers websites for matching vacancies. For example, to run the search every day at 8:00 AM and append the output to a log file:

```bash
0 8 * * * cd /absolute/path/to/project && /absolute/path/to/venv/bin/python cli.py --urls "https://catawiki.careers/vacancies,https://careers.adyen.com/" --keywords "Software Engineer" --location "Europe" --language "English" --headless "True" >> /absolute/path/to/jobfinder.log 2>&1
```

Tested with Python 3.11.
