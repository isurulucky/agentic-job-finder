# job-finder
Agentic app capable of navigating career portals to find suitable opportunities which match given criteria.


TODOs:
-----

1 TODO: remove convert to dictionaries 
    links = [
        {"url": cl.url, "text": cl.text}
        for cl in candidate_links
    ]

    # Filter using URL patterns and ATS domains
    links = find_candidate_urls(links)
   
1.1 Remove unwanted code

1.2 Python logging

2. Support local caching for sites successfully navigated?

3. Include self-learning capabilities for agents
   - Use a vector DB?
