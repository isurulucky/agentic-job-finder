# Agentic JobFinder

Agentic JobFinder is a command-line application that discovers job vacancies matching specified keywords, location, and language across multiple careers websites.

The application uses a multi-agent architecture, with specialized agents responsible for navigating careers websites and extracting vacancy URLs, and identifying vacancies that semantically match the search criteria. Deterministic filtering complements the agents to improve efficiency and reduce irrelevant results.

Playwright enables browser automation, while LangChain provides the LLM integration. Any compatible reasoning model supported by LangChain can be configured, allowing you to choose your preferred model provider.

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
- Optional `--model`: LLM model to use. Defaults to `gpt-5.4-mini`.
- Optional `--provider`: LLM provider. Defaults to `openai`.

### Example

```bash
python cli.py \
  --urls "https://www.coolblue.nl/en/vacancies,https://catawiki.careers/vacancies,https://careers.adyen.com/" \
  --keywords "Software Engineer" \
  --location "Europe" \
  --language "English"
```

Use a different model:

```bash 
python cli.py \
  --urls "https://careers.example.com" \
  --keywords "Python Developer" \
  --location "Netherlands" \
  --language "English" \
  --model "claude-sonnet-4-6" \
  --provider "anthropic"
```

## Running as a Cron Job

The application can be scheduled as a cron job to periodically search careers websites for matching vacancies. For example, to run the search every day at 8:00 AM and append the output to a log file:

```bash
0 8 * * * cd /absolute/path/to/project && /absolute/path/to/venv/bin/python cli.py --urls "https://www.coolblue.nl/en/vacancies,https://catawiki.careers/vacancies,https://careers.adyen.com/" --keywords "Software Engineer" --location "Europe" --language "English" >> /absolute/path/to/jobfinder.log 2>&1
```

Tested with Python 3.11.
