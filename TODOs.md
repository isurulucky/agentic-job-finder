# job-finder
Agentic app capable of navigating career portals to find suitable opportunities which match given criteria.


TODOs:
-----

1. Separate Browser interaction and finding matching vacancies. 
   - Change the api so that __init__ takes llm and details, and run takes target urls in brower agent
   - in filtering agent __init__ should take llm and stuff
   - test changes

1.1 Python logging

2. Support local caching for sites successfully navigated?

3. Include self-learning capabilities for agents
   - Use a vector DB?
