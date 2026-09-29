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
                const opacity = parseFloat(style.opacity);

                return (
                    rect.width > 0 &&
                    rect.height > 0 &&
                    style.display !== 'none' &&
                    style.visibility !== 'hidden' &&
                    opacity > 0.5 &&
                    style.pointerEvents !== 'none'
                );
            };

            const isBlockedInput = (el) => {
                const text = [
                    el.placeholder,
                    el.value,
                    el.name,
                    el.id,
                    el.getAttribute('aria-label'),
                    el.getAttribute('title')
                ]
                .filter(Boolean)
                .join(' ')
                .toLowerCase();

                return [
                    'cookie',
                    'privacy',
                    'newsletter',
                    'subscribe',
                    'consent'
                ].some(term => text.includes(term));
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

                    if (label) {
                        return label.innerText.trim();
                    }
                }

                const parentLabel = el.closest('label');

                if (parentLabel) {
                    return parentLabel.innerText.trim();
                }

                const ariaLabel = el.getAttribute('aria-label');

                if (ariaLabel) {
                    return ariaLabel.trim();
                }

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

                // Ignore hidden elements, including hidden backing
                // <select> elements used by custom dropdowns.
                if (!isVisible(el)) return;

                if (el.tagName === 'INPUT' &&isBlockedInput(el)) return;

                el.setAttribute('agent-id', idCounter);

                const tag = el.tagName.toLowerCase();

                const item = {
                    id: idCounter,
                    tag: tag,
                    role: el.getAttribute('role') || '',
                    text: getText(el),
                    label: getLabel(el),
                    type: el.type || '',
                    disabled: !!el.disabled
                };

                // Useful for navigation decisions
                if (tag === 'a') {
                    item.href = el.href;
                }

                // Useful for search/filter decisions
                if (tag === 'input') {

                    item.placeholder = el.placeholder || '';
                    item.value = el.value || '';

                    if (
                        el.type === 'checkbox' ||
                        el.type === 'radio'
                    ) {
                        item.checked = !!el.checked;
                    }
                }

                // Only visible native <select> elements reach this point.
                // Custom/hidden selects were filtered out by isVisible().
                if (tag === 'select') {

                    item.value = el.value || '';

                    item.options = Array.from(el.options)
                        .slice(0, 30)
                        .map(option => ({
                            text: option.text.trim(),
                            value: option.value,
                            selected: option.selected
                        }));
                }

                // Useful for custom dropdowns / expandable controls
                if (
                    tag === 'button' ||
                    el.getAttribute('role') === 'button' ||
                    el.getAttribute('role') === 'combobox'
                ) {
                    const expanded = el.getAttribute('aria-expanded');

                    if (expanded !== null) {
                        item.expanded = expanded === 'true';
                    }
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
    async def interact(
        state_version: int,
        agent_id: int,
        action: str,
        value: str = ""
    ) -> str:
        """
        Interacts with an element.

        - state_version: version of the state returned by get_page_state.
        - agent_id: exact integer ID from get_page_state.
        - action: 'click', 'fill', 'check', or 'select'.
        - value: text to type or option to select.
        """

        # Prevent using element IDs from an old page state.
        if state_version != globals()["state_version"]:
            return json.dumps({
                "success": False,
                "error": "STALE_STATE",
                "message": (
                    f"Element IDs belong to state {state_version}, "
                    f"but the current state is "
                    f"{globals()['state_version']}. "
                    "Call get_page_state before interacting."
                )
            })

        locator = page.locator(
            f"[agent-id='{agent_id}']"
        ).first

        try:
            # Make sure the element still exists.
            if await locator.count() == 0:
                return json.dumps({
                    "success": False,
                    "error": "ELEMENT_NOT_FOUND",
                    "message": (
                        f"Element {agent_id} was not found. "
                        "Call get_page_state again."
                    )
                })

            # Extract the target text for logging.
            element_text = await locator.evaluate(
                """
                (el) =>
                    el.innerText ||
                    el.placeholder ||
                    el.value ||
                    el.getAttribute('aria-label') ||
                    'No Text'
                """
            )

            element_text = (
                element_text
                .strip()
                [:100]
                .replace('\n', ' ')
            )

            print(
                f"\n[*] ACTION: {action.upper()} "
                f"| ID: {agent_id} "
                f"| TARGET: '{element_text}' "
                f"| INPUT: '{value}'"
            )

            prev_url = page.url

            # --------------------------------------------------
            # CLICK
            # --------------------------------------------------

            if action == "click":

                await locator.click()

            # --------------------------------------------------
            # FILL
            # --------------------------------------------------

            elif action == "fill":

                await locator.fill(value)
                await locator.press("Enter")

            # --------------------------------------------------
            # CHECK
            # --------------------------------------------------

            elif action == "check":

                await locator.check()

            # --------------------------------------------------
            # SELECT
            # --------------------------------------------------

            elif action == "select":

                tag_name = await locator.evaluate(
                    "(el) => el.tagName.toLowerCase()"
                )

                if tag_name != "select":
                    return json.dumps({
                        "success": False,
                        "error": "NOT_NATIVE_SELECT",
                        "message": (
                            "This is not a native select. "
                            "Click the visible dropdown, call "
                            "get_page_state, then click the desired option."
                        )
                    })

                is_visible = await locator.is_visible()

                if not is_visible:
                    return json.dumps({
                        "success": False,
                        "error": "HIDDEN_SELECT",
                        "message": (
                            "This select is hidden and is likely the backing "
                            "element for a custom dropdown. Do not use select. "
                            "Click the visible dropdown, call get_page_state, "
                            f"then click '{value}'."
                        )
                    })

                options = await locator.locator(
                    "option"
                ).all_text_contents()

                options = [option.strip() for option in options]

                if value not in options:
                    return json.dumps({
                        "success": False,
                        "error": "OPTION_NOT_FOUND",
                        "message": (
                            f"Option '{value}' was not found. "
                            f"Available options: {options}"
                        )
                    })

                await locator.select_option(label=value)

            # --------------------------------------------------
            # INVALID ACTION
            # --------------------------------------------------

            else:

                return json.dumps({
                    "success": False,
                    "error": "INVALID_ACTION",
                    "message": (
                        f"Unknown action '{action}'. "
                        "Use click, fill, check, or select."
                    )
                })

            # Wait briefly for navigation/dynamic updates.
            await page.wait_for_timeout(3000)

            # --------------------------------------------------
            # FEEDBACK
            # --------------------------------------------------

            current_url = page.url

            feedback = {
                "success": True,
                "action": action,
                "agent_id": agent_id,
                "message": (
                    f"Action '{action}' executed successfully "
                    f"on element {agent_id}."
                )
            }

            if current_url != prev_url:
                feedback["navigation"] = {
                    "from": prev_url,
                    "to": current_url
                }
            else:
                feedback["navigation"] = None
                feedback["message"] += (
                    " The URL did not change, but the page content "
                    "may have updated dynamically."
                )

            feedback["next_step"] = (
                "Call get_page_state before the next interaction."
            )

            return json.dumps(feedback)

        except Exception as e:

            error_msg = (
                f"Failed to perform {action} on {agent_id}: {str(e)}"
            )

            print(f"[!] {error_msg}")

            return json.dumps({
                "success": False,
                "error": "INTERACTION_FAILED",
                "message": error_msg,
                "next_step": (
                    "Call get_page_state and reassess the page."
                )
            })

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
            - Only enter search text on searchable text fields, not anything else. Never fill cookie, privacy, newsletter, login, or generic site-search inputs.
            - Only fill inputs whose label, placeholder, name, or surrounding context
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
