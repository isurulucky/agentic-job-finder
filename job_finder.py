import asyncio
import json
import time
from playwright.async_api import async_playwright
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool, tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from tools import BrowserTools

# os.environ["OPENAI_API_KEY"] = "your-api-key"

async def run_agent(target_url, keywords, location, language):    
    llm = ChatOpenAI(model="o4-mini", temperature=1)
    # llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    
    # Explicitly create the context to set standard browser locales/headers
    context = await browser.new_context()
    page = await context.new_page()
    
    print(f"[*] Navigating to {target_url}...")
    
    # Enforce the network idle directly in the goto command
    try:
        await page.goto(target_url, wait_until="networkidle", timeout=30000)
    except Exception as e:
        raise Exception(f"[*] Initial load error: {e}")

    # Instantiate BroweserTools
    browser_tools = BrowserTools(page, llm, keywords)

    # Bind tools to the LLM
    tools = [
        StructuredTool.from_function(
            coroutine=browser_tools.get_page_state,
            name="get_page_state",
        ),
        StructuredTool.from_function(
            coroutine=browser_tools.interact,
            name="interact",
        ),
        StructuredTool.from_function(
            coroutine=browser_tools.extract_urls,
            name="extract_urls",
        ),
    ]
    llm_with_tools = llm.bind_tools(tools)
    tools_map = {t.name: t for t in tools}

    # Agent loop
    messages = [
        SystemMessage(content=f"""
        You are an autonomous web agent finding job vacancies matching:
        - Keywords: "{keywords}"
        - Location: "{location}"
        - Language: "{language}"

        Use this loop: OBSERVE → DECIDE → ACT → VERIFY → repeat. Use as minimum loop iterations as possible.

        Rules:
        1. ALWAYS call `get_page_state` before acting.
        2. Remove consent forms, cookie banners, modals, and other blockers in each step.
        3. Only navigate career/job pages. Ignore generic sites, product, blog, documentation, or unrelated search pages.
        4. Prefer, in order:
            a) job opening/career related filters, such as drop downs, check boxes, radio buttons, buttons, etc.
            b) links to job-search/vacancies/current openings pages,
            c) career-specific search inputs.
            - Only enter search text on searchable text fields, not anything else. Never fill cookie, privacy, newsletter, login, or generic site-search inputs.
            - Only fill inputs whose label, placeholder, name, or surrounding context
        5. After EVERY `interact`, call `get_page_state` again. Do NOT call `interact` more than once in a row.
        6. Base each action on the current observed state. Avoid arbitrary clicks, repeated ineffective actions, and navigation loops.
        7. Treat keywords as semantic matches against job titles/descriptions and location as the actual vacancy location. Do not invent matches.
        8. Once matching vacancies are found, call `extract_urls`.
        9. If no matching vacancies can be found after exhausting relevant paths,
        return no URLs rather than inventing them.

        At every step, choose the single next action that makes the most progress
        toward finding matching vacancies.
        """),

        HumanMessage(content="Begin by observing the page state.")
    ]

    iterations = 15
    for iteration in range(0, iterations):
        print(f"\n--- Iteration {iteration+1}/{iterations} ---")
        
        # Ask the LLM to reason and pick a tool
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        # Check if the LLM decided to use a tool
        if not ai_msg.tool_calls:
            print("[*] Agent stopped calling tools. Text response:", ai_msg.content)
            break
            
        # Execute the chosen tool(s)
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            chosen_tool = tools_map[tool_name]
            tool_result = await chosen_tool.ainvoke(tool_args)
            
            # Append the result back to the conversation history
            messages.append(
                ToolMessage(
                    tool_call_id=tool_call["id"], 
                    content=json.dumps(tool_result)
                )
            )
            
            # Termination condition
            if tool_name == "extract_urls":
                print("[*] Extraction complete. Printing and terminating loop.")
                for url in tool_result:
                    print(url)
                await browser.close()
                await playwright.stop()
                return

            # Sleep 2 seconds
            time.sleep(2)

    print("[*] Loop limit reached or extraction finished.")
    await browser.close()
    await playwright.stop()

if __name__ == "__main__":
    target_url = "https://www.coolblue.nl/en/vacancies"
    # target_url = "https://www.leaseweb.com/en/about-us/career"
    # target_url = "https://catawiki.careers/vacancies"
    # target_url = "https://careers.adyen.com/"
    # target_url = "https://careers.bol.com/en/"
    # target_url = "https://www.personio.com/about-personio/careers/"
    keywords = "Software Engineer"
    location = "Europe"
    language = "English"
    asyncio.run(run_agent(target_url, keywords, location, language))
