import json
from pydantic import BaseModel, Field
from playwright.async_api import Page
from langchain_core.language_models.chat_models import BaseChatModel
from job_url_filter import filter_job_links

class JobLink(BaseModel):
    url: str = Field(description="The full URL of the job vacancy.")
    text: str = Field(description="The visible anchor text or title of the vacancy link.") 

class BrowserTools:
    def __init__(self, page: Page, llm: BaseChatModel, keywords: str):
        self.page = page
        self.llm = llm
        self.state_version = 0
        self.keywords = keywords
        self.state_valid = False

    async def get_page_state(self) -> dict:
        """Retrieves the current page state and assigns versioned element state IDs.

        Increments `self.state_version` and evaluates page state in the browser context,
        annotating visible interactive elements with a `state-id` DOM attribute.

        Returns:
        dict: A dictionary with the following structure:

        ```
            {
                "state_version": 1,
                "url": "https://example.com/page",
                "title": "Page Title",
                "elements": [
                    {
                    "state_id": 1,
                    "type": "button" | "dropdown" | "checkbox" | "radio" | "text_input" | "link",
                    "name": "Submit",
                    "disabled": false,
                    "href": "https://example.com/link",
                    "value": "",
                    "placeholder": "Enter email...",
                    "checked": false,
                    "options": [
                        {
                        "name": "Option Label",
                        "value": "opt_val",
                        "selected": true
                        }
                    ]
                    }
                ]
            }
            Element `state_id`s are only valid for the returned `state_version`.

            Type-specific properties:
            - `href`: present for links.
            - `value`, `placeholder`: present for text inputs.
            - `checked`: present for checkboxes and radio buttons.
            - `options`: present for dropdowns; contains at most 30 options.
        ```

        """

        self.state_version += 1
        self.state_valid = True

        print(
            f"\n[*] Retrieving page state "
            f"(version {self.state_version})"
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

        state = await self.page.evaluate(js_script)

        result = {
            "state_version": self.state_version,
            **state
        }

        return result

    async def interact(
        self,
        state_version: int,
        state_id: int,
        action: str,
        value: str = ""
    ) -> dict:
        """
        Interacts with an element from the current page state.

        `state_id` is only valid for the specified `state_version`.

        Args:
            state_version: Page state version associated with `state_id`.
            state_id: ID of the element to interact with.
            action: Action to perform: `click`, `fill`, `check`, or `select`.
            value: Value used by `fill` or `select`.

        Returns:
            dict: A dictionary with the following structure:

                {
                    "success": true,
                    "state_id": 1,
                    "action": "click",
                    "navigation": {
                        "from": "https://example.com",
                        "to": "https://example.com/next"
                    },
                    "error": null,
                    "message": null,
                    "next_step": "get_page_state"
                }

            On failure, `success` is false and `error`/`message` describe the
            failure. Fields not applicable to the outcome are null.

            Possible error codes include `STALE_STATE`, `ELEMENT_NOT_FOUND`,
            `NOT_NATIVE_SELECT`, `INVALID_ACTION`, and `INTERACTION_FAILED`.

            `next_step` instructs the agent which tool to call next and is
            always `get_page_state`.
        """

        # Reject stale state.
        if state_version != self.state_version or not self.state_valid:
            return {
                "success": False,
                "state_id": state_id,
                "action": action,
                "navigation": None,
                "error": "STALE_STATE",
                "message": (
                    f"State {state_version} is stale. "
                    f"Current state is {self.state_version}."
                ),
                "next_step": "get_page_state"
            }

        locator = self.page.locator(
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
                return {
                    "success": False,
                    "state_id": state_id,
                    "action": action,
                    "navigation": None,
                    "error": "ELEMENT_NOT_FOUND",
                    "message": (
                        f"State ID {state_id} was not found."
                    ),
                    "next_step": "get_page_state"
                }

            prev_url = self.page.url

            if action == "click":
                await locator.click(timeout=5000)

            elif action == "fill":
                await locator.fill(value, timeout=3000)
                await locator.press("Enter")

            elif action == "check":
                await locator.check(timeout=5000)

            elif action == "select":
                if await locator.evaluate(
                    "(el) => el.tagName.toLowerCase()"
                ) != "select":
                    return {
                        "success": False,
                        "state_id": state_id,
                        "action": action,
                        "navigation": None,
                        "error": "NOT_NATIVE_SELECT",
                        "message": (
                            "Only native dropdowns support 'select'."
                        ),
                        "next_step": "get_page_state"
                    }

                await locator.select_option(
                    label=value,
                    timeout=3000
                )

            else:
                return {
                    "success": False,
                    "state_id": state_id,
                    "action": action,
                    "navigation": None,
                    "error": "INVALID_ACTION",
                    "message": (
                        f"Unknown action '{action}'. "
                        "Use click, fill, check, or select."
                    ),
                    "next_step": "get_page_state"
                }

            current_url = self.page.url
            self.state_valid = False

            return {
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
                "error": None,
                "message": None,
                "next_step": "get_page_state"
            }

        except Exception as e:
            print(f"Error in interact: {str(e)} \n")

            return {
                "success": False,
                "state_id": state_id,
                "action": action,
                "navigation": None,
                "error": "INTERACTION_FAILED",
                "message": (
                    f"Failed to perform '{action}' "
                    f"on state_id {state_id}: {str(e)}"
                ),
                "next_step": "get_page_state"
            }
        
    async def extract_urls(self) -> list[JobLink]:
        """Extract all visible links from the current page with their text and URL.
        
            Returns:
                dict[str, str]: dictionary with with their URL (key) and visible text (value).
        """

        print("\n[*] Extracting links directly from DOM...")

        await self.page.wait_for_timeout(3000)

        js_extract = """
        () => {
            const links = Array.from(document.querySelectorAll('a[href]'));

            return links
                .map(a => ({
                    text: (a.innerText || a.getAttribute('aria-label') || '')
                        .trim()
                        .replace(/\\s+/g, ' '),
                    href: a.href
                }))
                .filter(item =>
                    item.text.length > 0 &&
                    item.href.startsWith('http')
                );
        }
        """
        raw_links = await self.page.evaluate(js_extract)

        # Deduplicate by URL
        unique_links = {}

        for link in raw_links:
            url = link["href"]
            text = link["text"]

            if url and url not in unique_links:
                unique_links[url] = text

        # Convert to dictionaries
        links = [
            {"url": url, "text": text}
            for url, text in unique_links.items()
        ]

        # Filter using URL patterns and ATS domains
        links = filter_job_links(links)

        # Convert to JobLink instances
        job_links = [
            JobLink(url=link["url"], text=link["text"]) for link in links
        ]

        print(f"[*] Extracted {len(job_links)} matching links.")

        return job_links

class MatchingVacancies(BaseModel):
    job_links: list[JobLink] = Field(
        description="All job vacancies matching the user's criteria. "
                    "Include every relevant vacancy, not just the best match."
    )

async def list_matching_vacancies(jobLinks: list[JobLink], llm: BaseChatModel, keywords: str) -> list[JobLink]:
        """Identifies job vacancy URLs matching the specified keywords.

            Args:
            jobLinks: List of unfiltered JobLink instances for discovered URLs.
            llm: Language model used to identify semantically matching vacancies.
            keywords: Keywords describing the desired vacancies.

            Returns:
            list[job_links]: List of JobLink instances corresponding to specific job vacancies that
            semantically match the keywords, with their visible link text.
            Returns an empty list if no matching vacancies are found.
        """

        if len(jobLinks) == 0:
            # No URLs passed
            print("Zero extracted URLs, no matching vacancies")
            return jobLinks

        # print(f"Input vacancy count: {len(jobLinks)}")

        # for job in jobLinks:
        #     print(f"INPUT: {job.url} | {job.text[:100]}")
        
        # Ask LLM to pick ONLY job vacancy URLs from the clean list
        extract_prompt = f"""
        Search criteria: {keywords}

        Candidate vacancies:
        {jobLinks}

        Return every vacancy related to software development.
        Return all matching vacancies in the job_links list using search criteria for direct keyword matches and contexual similarity.
        Do not return only the best match.
        """
        
        extractor = llm.with_structured_output(
            MatchingVacancies,
            method="json_schema",
        )        
        result = extractor.invoke(extract_prompt)

        return result.job_links
