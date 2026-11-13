# 🚀 Yet Another Agentic Job Hunter

### Because finding a job shouldn't feel like a full-time job.

Meet **Agentic Job Hunter**: your tireless, browser-wielding job-hunting sidekick. Give it some keywords, a location, and a language, point it at a bunch of careers websites, and let the agents do the legwork.

It explores careers pages, hunts down vacancy links, and figures out which jobs are actually relevant to your search. Less tab-hopping, more opportunities.

## 🧠 Not Just Another Scraper

Agentic Job Hunter uses a **multi-agent architecture**, where each component has a job of its own (pun absolutely intended).

- 🌐 **BrowserNavigationAgent** — explores careers websites and figures out how to navigate them.
- 🔍 **VacancyFilterAgent** — evaluates candidate vacancies and finds the ones that match your search criteria.
- ⚡ **Deterministic URL filtering** — narrows down candidate links before semantic filtering.

But here's where things get interesting...

## 🧬 It Learns. It Remembers. It Gets Faster.

Agentic Job Hunter features **experience-based self-learning**.

Every successful navigation strategy can be saved and reused. Experiences are associated with applicant tracking system (ATS) platforms and other careers website providers, so the application can recognize familiar websites and avoid rediscovering how to navigate them.

Here's how the magic happens:

1. 🧠 **Remember:** Retrieve previously stored navigation experiences for each target website.
2. 🤖 **Replay:** If an experience exists, `ExperienceBasedNavigator` attempts to replay it using deterministic browser automation.
3. 🔄 **Fallback:** If no experience exists, or replay fails, `BrowserNavigationAgent` takes over and explores the website using an LLM.
4. 💾 **Learn:** Successful agentic navigation experiences are stored for future use.
5. 🎯 **Filter:** Vacancy URLs are collected, filtered deterministically, and passed to `VacancyFilterAgent` for semantic matching.

### The Learning Loop

```text
                 🎯 Target URLs
                       |
              🧠 Retrieve Experiences
                       |
             +---------+---------+
             |                   |
      Experience Found      No Experience
             |                   |
   ExperienceBasedNavigator      |
             |                   |
       +-----+-----+             |
       |           |             |
    Success      Failure        |
       |           |             |
       |           +------+------+
       |                  |
       |       BrowserNavigationAgent
       |                  |
       +--------+---------+
                |
       ✅ Successful Results
                |
        💾 Store New Experiences
                |
        🔗 Collect Vacancy Links
                |
        ⚡ Deterministic URL Filter
                |
        🤖 VacancyFilterAgent
                |
        🎉 Matching Vacancies!
```

The navigation experiences live in a local **SQLite database**, so the application remembers what it learned even after it exits. No need to teach it the same website twice!

## 🛠️ Under the Hood

A few familiar tools make all this possible:

| Technology | What it does |
|---|---|
| 🐍 Python | Powers the application |
| 🎭 Playwright | Handles browser automation |
| 🦜 LangChain | Provides agent orchestration and LLM integration |
| 🗄️ SQLite | Stores navigation experiences |
| 🧠 OpenAI, Anthropic & Google Gemini | Provide the reasoning capabilities |

Multiple reasoning-model providers are supported through a unified LangChain interface.

The final matching vacancy URLs are printed to standard output.

## 🏁 Get Started

Ready to send your job-hunting sidekick to work?

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Generate an API key for your chosen model provider and set it as an environment variable.

For OpenAI:

```bash
export OPENAI_API_KEY="your-api-key"
```

For Anthropic:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### 3. Launch the job hunt

Navigate to the project directory and run:

```bash
python cli.py
```

## 🎛️ Command-Line Controls

Give your agent a few instructions before it heads out.

| Argument | Required? | Description | Example |
|---|---|---|---|
| `--urls` | ✅ Yes | Comma-separated careers or vacancy website URLs | `https://careers.example.com` |
| `--keywords` | ✅ Yes | Job titles or search keywords | `Software Engineer` |
| `--location` | ✅ Yes | Preferred job location | `Netherlands` |
| `--language` | ✅ Yes | Required vacancy language | `English` |
| `--headless` | ❌ No | Run the browser without displaying its UI | `True` |
| `--model` | ❌ No | Select the LLM model | `claude-sonnet-4-6` |
| `--provider` | ❌ No | Select the LLM provider | `anthropic` |

### Example: Find Software Engineering Jobs

```bash
python cli.py \
  --urls "https://www.coolblue.nl/en/vacancies,https://catawiki.careers/vacancies,https://careers.adyen.com/" \
  --keywords "Software Engineer" \
  --location "Netherlands" \
  --language "English"
```

### Example: Bring Your Own Brain 🧠

Want to use Anthropic instead?

```bash
python cli.py \
  --urls "https://careers.example.com" \
  --keywords "Python Developer" \
  --location "Europe" \
  --language "English" \
  --model "claude-sonnet-4-6" \
  --provider "anthropic"
```

Don't forget to set `ANTHROPIC_API_KEY` before running it.

## ⏰ Make It a Daily Habit

Why remember to look for jobs when your agent can do it for you?

You can schedule Agentic Job Hunter as a cron job and let it search for fresh opportunities every morning.

For example, to run it every day at **8:00 AM** and append the results to a log file:

```cron
0 8 * * * cd /absolute/path/to/project && /absolute/path/to/venv/bin/python cli.py --urls "https://catawiki.careers/vacancies,https://careers.adyen.com/" --keywords "Software Engineer" --location "Europe" --language "English" --headless "True" >> /absolute/path/to/JobHunter.log 2>&1
```

Replace the project and virtual environment paths with your own.

Now go make coffee. Your agent has a job to do. ☕

---

**Tested with Python 3.11.**

### Happy job hunting! 🎯

*May your searches be relevant, your navigation deterministic, and your vacancy list never-ending. And yet, may you not have to run this agent ever again!*
