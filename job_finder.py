import asyncio
import os
import json
import html2text
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# os.environ["OPENAI_API_KEY"] = "your-api-key"

class ExtractedJobs(BaseModel):
    job_urls: list[str] = Field(description="List of matching job vacancy URLs")

async def run_agent(target_url, keywords):    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Initialize Browser
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    page = await browser.new_page()
    
    print(f"[*] Navigating to {target_url}...")
    await page.goto(target_url)
    try:
        await page.wait_for_load_state("networkidle", timeout=5000)
    except Exception as e:
        raise e

    # Tools
    
    @tool
    async def get_page_state() -> str:
        """Returns the current interactive elements on the screen. Always call this first to see what you can interact with."""
        # JS snippet to inject 'agent-id' and build a clean tree of visible elements
        js_script = """
        () => {
            const elements = document.querySelectorAll('button, a, input, select, textarea, [role="button"], [role="checkbox"]');
            const tree = [];
            let idCounter = 1;
            elements.forEach(el => {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return; // Skip hidden
                
                el.setAttribute('agent-id', idCounter);
                tree.push({
                    id: idCounter,
                    tag: el.tagName.toLowerCase(),
                    text: (el.innerText || el.placeholder || el.value || el.getAttribute('aria-label') || '').substring(0, 60).trim(),
                    type: el.type || el.getAttribute('role') || ''
                });
                idCounter++;
            });
            return JSON.stringify(tree, null, 2);
        }
        """
        tree_json = await page.evaluate(js_script)
        return f"Current Interactive Elements:\n{tree_json}"

    @tool
    async def interact(agent_id: int, action: str, value: str = "") -> str:
        """
        Interacts with an element. 
        - agent_id: The exact integer ID of the element from get_page_state.
        - action: Must be one of 'click', 'fill', 'check', 'select'.
        - value: The text to type (for 'fill') or option to choose (for 'select'). Empty for click/check.
        """
        print(f"[*] Agent executing: {action} on element ID {agent_id} with value '{value}'")
        locator = page.locator(f"[agent-id='{agent_id}']").first
        
        try:
            if action == "click":
                await locator.click()
            elif action == "fill":
                await locator.fill(value)
                await locator.press("Enter")
            elif action == "check":
                await locator.check()
            elif action == "select":
                await locator.select_option(label=value)
            else:
                raise ValueError(f"unknown action: {action}")
            
            # Wait for any resulting network activity or DOM changes
            await asyncio.sleep(2)
            return f"Successfully performed {action} on element {agent_id}."
        except Exception as e:
            return f"Failed to perform {action} on {agent_id}: {str(e)}"

    @tool
    async def extract_and_print_urls() -> str:
        """Extracts job URLs from the current page matching the user's keywords and prints them to the console. Call this ONLY when you see job listings."""
        print("\n[*] Tool called: Extracting DOM to Markdown...")
        
        html_converter = html2text.HTML2Text()
        html_converter.ignore_links = False
        html_converter.ignore_images = True
        
        html_content = await page.content()
        markdown = html_converter.handle(html_content)[:15000] # Truncate for safety
        
        extract_prompt = f"Extract matching job URLs for keywords: '{keywords}' from this markdown:\n\n{markdown}"
        extractor = llm.with_structured_output(ExtractedJobs)
        result = await extractor.ainvoke(extract_prompt)
        
        print("\n=== FINAL EXTRACTED JOB URLS ===")
        for url in result.job_urls:
            print(url)
            
        return "SUCCESS_EXTRACTED_AND_PRINTED"

    # Bind tools to the LLM
    tools = [get_page_state, interact, extract_and_print_urls]
    llm_with_tools = llm.bind_tools(tools)
    tools_map = {t.name: t for t in tools}

    # Agent loop
    
    messages = [
        SystemMessage(content="You are a job search automation agent. Your goal is to navigate career portals, handle cookie banners, search/filter for jobs, and finally extract URLs of opportunities which match given keywords. Use tools to observe and act."),
        HumanMessage(content=f"Find jobs matching: {keywords}. First, observe the page state.")
    ]

    for iteration in range(1, 11):
        print(f"\n--- Iteration {iteration}/10 ---")
        
        # 1. Ask the LLM to reason and pick a tool
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        
        # 2. Check if the LLM decided to use a tool
        if not ai_msg.tool_calls:
            print("[*] Agent stopped calling tools. Text response:", ai_msg.content)
            break
            
        # 3. Execute the chosen tool(s)
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            chosen_tool = tools_map[tool_name]
            tool_result = await chosen_tool.ainvoke(tool_args)
            
            # Append the result back to the conversation history
            messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_result)))
            
            # 4. Termination condition
            if tool_name == "extract_and_print_urls":
                print("[*] Extraction complete. Terminating loop.")
                await browser.close()
                await playwright.stop()
                return

    print("[*] Loop limit reached or extraction finished.")
    await browser.close()
    await playwright.stop()

if __name__ == "__main__":
    target_url = "https://www.coolblue.nl/en/vacancies"
    keywords = "data science, machine learning, python"
    asyncio.run(run_agent(target_url, keywords))
