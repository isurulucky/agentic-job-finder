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
        observation.

        Element state_ids are only valid for this state_version.
        Always call this before interacting with the page.
        """

        global state_version

        state_version += 1

        print(
            f"\n[*] Retrieving page state "
            f"(version {state_version})"
        )

        js_script = """
        () => {

            const isVisible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                return (
                    rect.width > 0 &&
                    rect.height > 0 &&
                    style.display !== 'none' &&
                    style.visibility !== 'hidden' &&
                    parseFloat(style.opacity) > 0.5 &&
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

            const getName = (el) => {
                // Explicit label
                if (el.id) {
                    const label = document.querySelector(
                        `label[for="${CSS.escape(el.id)}"]`
                    );

                    if (label) {
                        const text = label.innerText
                            .replace(/\\s+/g, ' ')
                            .trim();

                        if (text) return text;
                    }
                }

                // Label wrapping the element
                const parentLabel = el.closest('label');

                if (parentLabel) {
                    const text = parentLabel.innerText
                        .replace(/\\s+/g, ' ')
                        .trim();

                    if (text) return text;
                }

                // Accessible name
                const ariaLabel = el.getAttribute('aria-label');

                if (ariaLabel) {
                    return ariaLabel.trim();
                }

                // Title
                const title = el.getAttribute('title');

                if (title) {
                    return title.trim();
                }

                // Placeholder for text inputs
                if (el.placeholder) {
                    return el.placeholder.trim();
                }

                // Visible text
                const text = el.innerText
                    .replace(/\\s+/g, ' ')
                    .trim();

                if (text) {
                    return text.substring(0, 120);
                }

                // Input value as final fallback
                if (el.value) {
                    return String(el.value)
                        .replace(/\\s+/g, ' ')
                        .trim()
                        .substring(0, 120);
                }

                return '';
            };

            const interactive = [];

            let stateId = 1;

            const addElement = (el, type, extra = {}) => {
                if (!isVisible(el)) return;

                el.setAttribute('state-id', stateId);

                interactive.push({
                    state_id: stateId,
                    type: type,
                    name: getName(el),
                    disabled: !!el.disabled,
                    ...extra
                });

                stateId++;
            };

            /*
            * 1. Buttons
            *
            * Includes native buttons and elements using role="button".
            * Cookie/consent buttons remain visible.
            */
            document.querySelectorAll(
                'button, [role="button"]'
            ).forEach(el => {
                addElement(el, 'button');
            });

            /*
            * 2. Native dropdowns
            *
            * Only visible <select> elements are exposed.
            * Hidden/transparent backing selects for custom dropdowns
            * are ignored.
            */
            document.querySelectorAll('select').forEach(el => {
                if (!isVisible(el)) return;

                addElement(el, 'dropdown', {
                    options: Array.from(el.options)
                        .slice(0, 30)
                        .map(option => ({
                            name: option.text.trim(),
                            value: option.value,
                            selected: option.selected
                        }))
                });
            });

            /*
            * 3. Checkboxes
            */
            document.querySelectorAll(
                'input[type="checkbox"], [role="checkbox"]'
            ).forEach(el => {
                if (
                    el.tagName === 'INPUT' &&
                    isBlockedInput(el)
                ) {
                    return;
                }

                addElement(el, 'checkbox', {
                    checked: !!el.checked
                });
            });

            /*
            * 4. Radio buttons
            */
            document.querySelectorAll(
                'input[type="radio"], [role="radio"]'
            ).forEach(el => {
                addElement(el, 'radio', {
                    checked: !!el.checked
                });
            });

            /*
            * 5. Text inputs
            *
            * Only inputs intended for textual entry are exposed.
            */
            document.querySelectorAll(
                'input, textarea'
            ).forEach(el => {

                const tag = el.tagName.toLowerCase();
                const inputType = (el.type || 'text').toLowerCase();

                const isTextInput =
                    tag === 'textarea' ||
                    [
                        'text',
                        'search',
                        'email',
                        'tel',
                        'url'
                    ].includes(inputType);

                if (!isTextInput) return;

                // Do not expose cookie/privacy/newsletter inputs.
                if (tag === 'input' && isBlockedInput(el)) {
                    return;
                }

                addElement(el, 'text_input', {
                    value: el.value || '',
                    placeholder: el.placeholder || ''
                });
            });

            /*
            * 6. Links
            */
            document.querySelectorAll('a').forEach(el => {
                if (!el.href) return;

                addElement(el, 'link', {
                    href: el.href
                });
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
        state_id: int,
        action: str,
        value: str = ""
    ) -> str:
        """
        Interact with an element from the current page state.

        state_id is only valid for the specified state_version.
        Actions:
        - click
        - fill
        - check
        - select
        """

        # Reject stale state.
        if state_version != globals()["state_version"]:
            return json.dumps({
                "success": False,
                "error": "STALE_STATE",
                "message": (
                    f"State {state_version} is stale. "
                    f"Current state is {globals()['state_version']}. "
                    "Call get_page_state first."
                )
            })

        locator = page.locator(
            f"[state-id='{state_id}']"
        ).first

        target = await locator.evaluate("""
        (el) => {
            const label = el.id
                ? document.querySelector(
                    `label[for="${CSS.escape(el.id)}"]`
                )
                : null;

            return (
                label?.innerText ||
                el.closest('label')?.innerText ||
                el.getAttribute('aria-label') ||
                el.getAttribute('title') ||
                el.placeholder ||
                el.innerText ||
                el.value ||
                ''
            )
            .replace(/\\s+/g, ' ')
            .trim()
            .substring(0, 120);
        }
        """)

        print(
            f"\n[*] Interacting "
            f"| state={state_version}:{state_id} "
            f"| action={action.upper()} "
            f"| target='{target}' "
            f"| value='{value}'"
        )

        try:
            if await locator.count() == 0 or not await locator.is_visible():
                return json.dumps({
                    "success": False,
                    "error": "ELEMENT_NOT_FOUND",
                    "message": (
                        f"State ID {state_id} was not found. "
                        "Call get_page_state again."
                    )
                })

            prev_url = page.url

            if action == "click":
                await locator.click(timeout=3000)

            elif action == "fill":
                await locator.fill(value, timeout=3000)
                await locator.press("Enter")

            elif action == "check":
                await locator.check(timeout=3000)

            elif action == "select":
                if await locator.evaluate(
                    "(el) => el.tagName.toLowerCase()"
                ) != "select":
                    return json.dumps({
                        "success": False,
                        "error": "NOT_NATIVE_SELECT",
                        "message": (
                            "Only native dropdowns support 'select'. "
                            "For a custom dropdown, click it, then "
                            "call get_page_state and click the option."
                        )
                    })

                await locator.select_option(label=value, timeout=3000)

            else:
                return json.dumps({
                    "success": False,
                    "error": "INVALID_ACTION",
                    "message": (
                        f"Unknown action '{action}'. "
                        "Use click, fill, check, or select."
                    )
                })

            current_url = page.url

            return json.dumps({
                "success": True,
                "state_id": state_id,
                "action": action,
                "navigation": (
                    {
                        "from": prev_url,
                        "to": current_url
                    }
                    if current_url != prev_url
                    else None
                ),
                "next_step": (
                    "Call get_page_state before the next call to interact."
                )
            })

        except Exception as e:
            print(f"Error in interact: {str(e)} \n")
            return json.dumps({
                "success": False,
                "error": "INTERACTION_FAILED",
                "message": (
                    f"Failed to perform '{action}' "
                    f"on state_id {state_id}: {str(e)}"
                ),
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
            a) job opening/career related filters, such as drop downs, check boxes, radio buttons, buttons, etc.
            b) links to job-search/vacancies/current openings pages,
            c) career-specific search inputs.
            - Only enter search text on searchable text fields, not anything else. Never fill cookie, privacy, newsletter, login, or generic site-search inputs.
            - Only fill inputs whose label, placeholder, name, or surrounding context
        5. After EVERY `interact`, call `get_page_state` again. Do NOT call `interact` more than once in a row.
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
