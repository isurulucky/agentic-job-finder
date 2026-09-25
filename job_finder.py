import asyncio
import os
import json
import time
import html2text
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# os.environ["OPENAI_API_KEY"] = "your-api-key"
state_version = 0

class ExtractedJobs(BaseModel):
    job_urls: list[str] = Field(description="List of matching job vacancy URLs")

async def run_agent(target_url, keywords, location):    
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

    # Tools 
    # @tool
    # async def get_page_state() -> str:
    #     """Returns the current interactive elements on the screen. Always call this first to see what you can interact with."""

    #     # JS snippet to inject 'agent-id' and build a clean tree of visible elements
    #     print("\n[*] Retrieving current page state")

    #     js_script = """
    #     () => {
    #         const elements = document.querySelectorAll('button, a, input, select, textarea, [role="button"], [role="checkbox"]');
    #         const tree = [];
    #         let idCounter = 1;
    #         elements.forEach(el => {
    #             const rect = el.getBoundingClientRect();
    #             if (rect.width === 0 || rect.height === 0) return; // Skip hidden
                
    #             el.setAttribute('agent-id', idCounter);
    #             tree.push({
    #                 id: idCounter,
    #                 tag: el.tagName.toLowerCase(),
    #                 text: (el.innerText || el.placeholder || el.value || el.getAttribute('aria-label') || '').substring(0, 60).trim(),
    #                 type: el.type || el.getAttribute('role') || ''
    #             });
    #             idCounter++;
    #         });
    #         return JSON.stringify(tree, null, 2);
    #     }
    #     """
    #     tree_json = await page.evaluate(js_script)
    #     return f"Current Interactive Elements:\n{tree_json}"

    # global state version to track ids against.

    @tool
    async def get_page_state() -> str:
        """
        Returns the current page state and a state_version identifying this
        observation. Element IDs are only valid for this state_version.
        Always call this before interacting with the page.
        """

        global state_version
        state_version += 1

        print(f"\n[*] Retrieving page state (version {state_version})")

        js_script = """
        () => {
            const isVisible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                return (
                    rect.width > 0 &&
                    rect.height > 0 &&
                    style.display !== 'none' &&
                    style.visibility !== 'hidden'
                );
            };

            const getText = (el) => {
                return (
                    el.innerText ||
                    el.getAttribute('aria-label') ||
                    el.getAttribute('title') ||
                    el.placeholder ||
                    el.value ||
                    ''
                )
                .replace(/\\s+/g, ' ')
                .trim()
                .substring(0, 120);
            };

            const getLabel = (el) => {
                if (el.id) {
                    const label = document.querySelector(
                        `label[for="${CSS.escape(el.id)}"]`
                    );
                    if (label) return label.innerText.trim();
                }

                const parentLabel = el.closest('label');
                if (parentLabel) return parentLabel.innerText.trim();

                return '';
            };

            const elements = document.querySelectorAll(
                'a, button, input, select, textarea, ' +
                '[role="button"], [role="checkbox"], [role="radio"], ' +
                '[role="combobox"]'
            );

            const interactive = [];
            let idCounter = 1;

            elements.forEach(el => {
                if (!isVisible(el)) return;

                el.setAttribute('agent-id', idCounter);

                const item = {
                    id: idCounter,
                    tag: el.tagName.toLowerCase(),
                    role: el.getAttribute('role') || '',
                    text: getText(el),
                    label: getLabel(el),
                    type: el.type || '',
                    disabled: !!el.disabled
                };

                // Useful for navigation decisions
                if (el.tagName === 'A') {
                    item.href = el.href;
                }

                // Useful for search/filter decisions
                if (el.tagName === 'INPUT') {
                    item.placeholder = el.placeholder || '';
                    item.value = el.value || '';
                    item.checked = !!el.checked;
                }

                // Expose available options so the agent doesn't need to guess
                if (el.tagName === 'SELECT') {
                    item.value = el.value || '';

                    item.options = Array.from(el.options)
                        .slice(0, 30)
                        .map(option => ({
                            text: option.text.trim(),
                            value: option.value,
                            selected: option.selected
                        }));
                }

                if (
                    el.tagName === 'BUTTON' ||
                    el.getAttribute('role') === 'button'
                ) {
                    item.expanded = el.getAttribute('aria-expanded');
                }

                interactive.push(item);
                idCounter++;
            });

            return {
                url: window.location.href,
                title: document.title,
                elements: interactive
            };
        }
        """

        state = await page.evaluate(js_script)

        result = {
            "state_version": state_version,
            **state
        }

        return json.dumps(result, indent=2)

    @tool
    async def interact(state_version: int, agent_id: int, action: str, value: str = "") -> str:
        """
        Interacts with an element. 
        - state_version: version of the state as determined by 'get_page_state'. 
        - agent_id: The exact integer ID of the element from get_page_state.
        - action: Must be one of 'click', 'fill', 'check', 'select'.
        - value: The text to type (for 'fill') or option to choose (for 'select'). Empty for click/check.
        """
        if state_version != globals()["state_version"]:
            return json.dumps({
                "success": False,
                "error": "STALE_STATE",
                "message": (
                    f"Element IDs belong to state {state_version}, "
                    f"but the current state is {globals()['state_version']}. "
                    "Call get_page_state before interacting."
                )
            })

        locator = page.locator(f"[agent-id='{agent_id}']").first
        
        try:
            # Extract the text of the element the LLM is targeting
            element_text = await locator.evaluate("(el) => el.innerText || el.placeholder || el.value || el.getAttribute('aria-label') || 'No Text'")
            element_text = element_text.strip()[:100].replace('\n', ' ')
            
            print(f"\n[*] ACTION: {action.upper()} | ID: {agent_id} | TARGET: '{element_text}' | INPUT: '{value}'")
            
            prev_url = page.url
            
            # Execute action
            if action == "click":
                await locator.click()
            elif action == "fill":
                await locator.fill(value)
                await locator.press("Enter")
            elif action == "check":
                await locator.check()
            elif action == "select":
                await locator.select_option(label=value)
            
            # Wait for network requests or navigation to settle
            await page.wait_for_timeout(3000) 
            
            # LLM Feedback 
            current_url = page.url
            feedback = f"Action '{action}' executed on element {agent_id}."
            
            if current_url != prev_url:
                feedback += f" The browser navigated to a new URL: {current_url}."
            else:
                feedback += " The URL did not change, but the page content may have updated dynamically."
                
            feedback += " You MUST call get_page_state next to observe the new elements."
            return feedback
            
        except Exception as e:
            error_msg = f"Failed to perform {action} on {agent_id}: {str(e)}"
            print(f"[!] {error_msg}")
            return error_msg

    @tool
    async def extract_and_print_urls() -> str:
        """Extracts job URLs from the current page matching the user's keywords and prints them to the console."""
        print("\n[*] Tool called: Extracting job links directly from DOM...")
        
        # Wait a moment for dynamic job cards to render in React/Vue
        await page.wait_for_timeout(2000)
        
        # Extract all visible links with their inner text and full absolute hrefs
        js_extract = """
        () => {
            const links = Array.from(document.querySelectorAll('a[href]'));
            return links
                .map(a => ({
                    text: (a.innerText || a.getAttribute('aria-label') || '').trim().replace(/\\s+/g, ' '),
                    href: a.href // Absolute URL
                }))
                .filter(item => item.text.length > 0 && item.href.startsWith('http'));
        }
        """
        raw_links = await page.evaluate(js_extract)
        
        # Deduplicate links
        unique_links = {link['href']: link['text'] for link in raw_links}
        formatted_links = [f"Text: '{text}' | URL: {href}" for href, text in unique_links.items()]
        
        # Truncate string representation if page has hundreds of nav links
        links_text = "\n".join(formatted_links[:150]) 
        
        # Ask LLM to pick ONLY job vacancy URLs from the clean list
        extract_prompt = f"""
        Given this list of links on a career page, extract ONLY the URLs that correspond to specific job vacancies matching keywords: '{keywords}'.
        Ignore generic navigation links (like 'About Us', 'Privacy', 'Categories').
        
        Links:
        {links_text}
        """
        
        extractor = llm.with_structured_output(ExtractedJobs)
        result = extractor.invoke(extract_prompt)
        
        print("\n=== FINAL EXTRACTED JOB URLS ===")
        for url in result.job_urls:
            print(url)
            
        return "SUCCESS_EXTRACTED_AND_PRINTED"

    # @tool
    # async def extract_and_print_urls() -> str:
    #     """Extracts job URLs from the current page matching the user's keywords and prints them to the console. Call this ONLY when you see job listings."""
    #     print("\n[*] Tool called: Extracting DOM to Markdown...")
        
    #     html_converter = html2text.HTML2Text()
    #     html_converter.ignore_links = False
    #     html_converter.ignore_images = True
        
    #     html_content = await page.content()
    #     markdown = html_converter.handle(html_content)[:15000] # Truncate for safety
        
    #     extract_prompt = f"Extract matching job URLs for keywords: '{keywords}' from this markdown:\n\n{markdown}"
    #     extractor = llm.with_structured_output(ExtractedJobs)
    #     result = await extractor.ainvoke(extract_prompt)
        
    #     print("\n=== FINAL EXTRACTED JOB URLS ===")
    #     for url in result.job_urls:
    #         print(url)
            
    #     return "SUCCESS_EXTRACTED_AND_PRINTED"

    # Bind tools to the LLM
    tools = [get_page_state, interact, extract_and_print_urls]
    llm_with_tools = llm.bind_tools(tools)
    tools_map = {t.name: t for t in tools}

    # Agent loop
    messages = [
        SystemMessage(content=f"""
        You are an autonomous web agent finding job vacancies matching:
        - Keywords: "{keywords}"
        - Location: "{location}"

        Use this loop: OBSERVE → DECIDE → ACT → VERIFY → repeat.

        Rules:
        1. ALWAYS call `get_page_state` before acting.
        2. Remove consent forms, cookie banners, modals, and other blockers in each step.
        3. Only navigate career/job pages. Ignore generic sites, product, blog, documentation, or unrelated search pages.
        4. Prefer, in order:
            a) relevant career filters,
            b) links to job-search/vacancy pages,
            c) clearly career-specific search inputs.
            Only enter search text on searchable text fields, not anything else.
            Never use generic site search.
        5. After EVERY `interact`, call `get_page_state` again. Never assume an action succeeded.
        6. Base each action on the current observed state. Avoid arbitrary clicks, repeated ineffective actions, and navigation loops.
        7. Treat keywords as semantic matches against job titles/descriptions and location as the actual vacancy location. Do not invent matches.
        8. Once matching vacancies are found, call `extract_and_print_urls`.
        9. If no matching vacancies can be found after exhausting relevant paths,
        return no URLs rather than inventing them.

        At every step, choose the single next action that makes the most progress
        toward finding matching vacancies.
        """),

        HumanMessage(content="Begin by observing the page state.")
    ]

    for iteration in range(1, 11):
        print(f"\n--- Iteration {iteration}/10 ---")
        
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
            messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(tool_result)))
            
            # Termination condition
            if tool_name == "extract_and_print_urls":
                print("[*] Extraction complete. Terminating loop.")
                await browser.close()
                await playwright.stop()
                return

            # Sleep 2 seconds
            time.sleep(3)

    print("[*] Loop limit reached or extraction finished.")
    await browser.close()
    await playwright.stop()

if __name__ == "__main__":
    # target_url = "https://www.coolblue.nl/en/vacancies"
    target_url = "https://www.leaseweb.com/en/about-us/career"
    # target_url = "https://catawiki.careers/vacancies"
    keywords = "Backend Software Engineering"
    location = "Europe"
    asyncio.run(run_agent(target_url, keywords, location))
