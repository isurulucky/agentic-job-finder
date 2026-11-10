# Agentic JobFinder

Agentic JobFinder is a command-line application that discovers job vacancies matching specified keywords, location, and language across multiple careers websites.

The application uses a **multi-agent architecture** with specialized agents for navigating careers websites, extracting vacancy URLs, and identifying vacancies that semantically match the search criteria. Deterministic filtering complements the agents to improve efficiency and reduce irrelevant results.

A key feature is **experience-based self-learning**, which enables the application to learn from and reuse previously successful browser navigation strategies. Navigation experiences are stored and associated with applicant tracking system (ATS) platforms and other careers website providers. When revisiting a familiar platform, the application first attempts to replay a stored experience through deterministic browser automation. This avoids unnecessary LLM calls for websites that can be navigated using previously discovered strategies.

When no suitable experience is available, or deterministic replay fails, the application falls back to its LLM-powered navigation agent to explore the website and discover a new navigation strategy. Successful experiences are then stored for future reuse. This creates a continuous learning cycle in which the application becomes increasingly efficient as it encounters more careers websites.

The interplay between experience-based deterministic navigation and LLM-powered agentic navigation can be illustrated as follows:

                    Target URLs
                         |
                  Retrieve Experiences
                         |
              +----------+-----------+
              |                      |
       Experience Found       No Experience
              |                      |
    ExperienceBasedNavigator         |
              |                      |
       +------+-------+              |
       |              |              |
    Success         Failure          |
       |              |              |
       |              +-------+------+
       |                      |
       |             BrowserNavigationAgent
       |                      |
       +----------+-----------+
                  |
          Successful Results
                  |
          Store Experiences
                  |
          Collect Vacancy Links
                  |
          Deterministic URL Filter
                  |
          VacancyFilterAgent
                  |
          Final Matching Vacancies


Navigation experiences are currently stored in a local SQLite database, allowing the application to retain and reuse learned strategies across separate runs.

Playwright powers browser automation, while LangChain provides agent orchestration and LLM integration. The current implementation supports multiple reasoning-model providers, including OpenAI, Anthropic, and Google Gemini, through a unified LangChain interface.

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
  --location "Netherlands" \
  --language "English"
```

Use a different model:

```bash 
python cli.py \
  --urls "https://careers.example.com" \
  --keywords "Python Developer" \
  --location "Europe" \
  --language "English" \
  --model "claude-sonnet-4-6" \
  --provider "anthropic"
```

This requires environment variable `ANTHROPIC_API_KEY` to be set. 

## Running as a Cron Job

The application can be scheduled as a cron job to periodically search careers websites for matching vacancies. For example, to run the search every day at 8:00 AM and append the output to a log file:

```bash
0 8 * * * cd /absolute/path/to/project && /absolute/path/to/venv/bin/python cli.py --urls "https://catawiki.careers/vacancies,https://careers.adyen.com/" --keywords "Software Engineer" --location "Europe" --language "English" --headless "True" >> /absolute/path/to/jobfinder.log 2>&1
```

Tested with Python 3.11.
