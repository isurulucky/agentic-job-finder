import asyncio
import json

from playwright.async_api import async_playwright

from langchain.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from tools import BrowserTools, JobLink
from nav_result import WebsiteNavigationResult, normalize_job_links


def _serialize_default(obj: object) -> object:
    """Serialize supported objects into JSON-compatible values."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()

    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )

class BrowserNavigationAgent:
    """Navigate careers websites and extract candidate vacancy URLs."""
    def __init__(
        self,
        keywords: str,
        location: str,
        language: str,
        llm:  BaseChatModel,
        max_iterations: int = 15,
        navigation_timeout_millis: int = 30_000

    ) -> None:
        self.keywords = keywords
        self.location = location
        self.language = language
        self.llm = llm
        self.max_iterations = max_iterations
        self.navigation_timeout_millis = navigation_timeout_millis
        self._page_state_tool = "get_page_state"
        self._interact_tool = "interact"
        self._extract_urls_tool = "extract_urls"

    def _build_system_prompt(self, experiences) -> str:
        return f"""
        You are an autonomous browser agent responsible for discovering
        job vacancy URLs on careers websites.

        Filtering criteria:
        - Keywords: "{self.keywords}"
        - Location: "{self.location}"
        - Language: "{self.language}"

        Use the following approach:

        1. OBSERVE: Always call `{self._page_state_tool}` before deciding what to do.
        2. DECIDE: Choose the best action based on the current page state and search criteria.
        3. ACT: Use `{self._interact_tool}` to navigate the website or apply relevant job-search filters. Avoid unrelated inputs and interactions.
        4. VERIFY: After every `{self._interact_tool}` call, call `{self._page_state_tool}` to verify the outcome. Never call `{self._interact_tool}` twice consecutively.

        Navigation guidelines:
        - Focus on careers pages, job listings, and vacancy search interfaces.
        - Dismiss cookie banners, consent forms, and other blockers when necessary.
        - Prefer job-related filters and vacancy links over unnecessary navigation.
        - Avoid unrelated pages, arbitrary clicks, repeated ineffective actions, and navigation loops.
        - Use the search criteria to guide navigation and available filtering.
        - Filling a search field or filter does not necessarily apply its value. 
        After entering a value, inspect the updated page state and click the 
        associated Search, Submit, Apply, or similar button when required.
        - For dropdowns and autocomplete fields, select the most suitable option.
        If necessary click on the suitable option. Or if an Apply, Done, Confirm or similar button is available, click it to apply the selection.
        - If a dropdown does not contain an exact match for the filtering criteria,
        choose the most suitable option (e.g. a town or area instead of the country).
        - Verify that searches and filters have actually been applied before proceeding.

        Once navigation and filtering are complete, collect candidate vacancy URLs
        and their visible link text by calling `{self._extract_urls_tool}`.
        """

    async def run(
        self,
        target_urls: list[str],
        experiences: dict[str, list[dict]] | None = None,
        headless: bool = False,
    ) -> list[WebsiteNavigationResult]:
        """Navigate target websites and return per-website navigation results."""

        navigation_results: list[WebsiteNavigationResult] = []

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=headless)

            try:
                for target_url in target_urls:
                    context = None
                    termination_reason = "iteration_limit"
                    site_candidates: dict[str, JobLink] = {}
                    navigation_steps: list[dict] = []

                    # Tracks the most recently retrieved page state.
                    current_page_state = {}

                    try:
                        context = await browser.new_context()
                        page = await context.new_page()

                        print(f"\n[*] Navigating to {target_url}...")

                        try:
                            await page.goto(
                                target_url,
                                wait_until="domcontentloaded",
                                timeout=self.navigation_timeout_millis,
                            )
                            await page.wait_for_timeout(3000)

                        except Exception as exc:
                            termination_reason = "navigation_error"

                            print(
                                f"[!] Page load error for "
                                f"{target_url}: {exc}"
                            )

                            navigation_results.append(
                                WebsiteNavigationResult(
                                    url=target_url,
                                    termination_reason=termination_reason,
                                    experience={
                                        "target_url": target_url,
                                        "steps": [],
                                        "termination_reason": termination_reason,
                                        "result_count": 0,
                                    },
                                )
                            )
                            continue

                        browser_tools = BrowserTools(page)

                        # This agent handles navigation and URL extraction only.
                        tools = [
                            StructuredTool.from_function(
                                coroutine=browser_tools.get_page_state,
                                name=self._page_state_tool,
                            ),
                            StructuredTool.from_function(
                                coroutine=browser_tools.interact,
                                name=self._interact_tool,
                            ),
                            StructuredTool.from_function(
                                coroutine=browser_tools.extract_urls,
                                name=self._extract_urls_tool,
                            ),
                        ]

                        llm_with_tools = self.llm.bind_tools(tools)
                        tools_map = {
                            tool.name: tool
                            for tool in tools
                        }

                        known_experiences = (experiences or {}).get(
                            target_url,
                            [],
                        )

                        print(
                            f"\n[*] Found {len(known_experiences)} "
                            "previous navigation experiences."
                        )

                        if known_experiences:
                            print(
                                "Previous nav experience #1: "
                                f"{known_experiences[0]}"
                            )

                        messages = [
                            SystemMessage(
                                content=self._build_system_prompt(
                                    known_experiences
                                )
                            ),
                            HumanMessage(
                                content=(
                                    "Begin by observing the page state "
                                    f"at {target_url}."
                                )
                            ),
                        ]

                        extraction_completed = False

                        for iteration in range(self.max_iterations):
                            print(
                                f"\n--- Iteration "
                                f"{iteration + 1}/{self.max_iterations} ---"
                            )

                            try:
                                ai_msg = await llm_with_tools.ainvoke(messages)

                            except Exception as exc:
                                termination_reason = "model_error"

                                print(
                                    f"[!] Model invocation failed for "
                                    f"{target_url}: {exc}"
                                )
                                break

                            messages.append(ai_msg)

                            if not ai_msg.tool_calls:
                                termination_reason = "agent_stopped"

                                print(
                                    "[*] Agent stopped calling tools. "
                                    f"Response: {ai_msg.content}"
                                )
                                break

                            for tool_call in ai_msg.tool_calls:
                                tool_call_id = tool_call["id"]
                                tool_name = tool_call["name"]
                                tool_args = tool_call.get("args", {})

                                # Ensure every tool call receives a result,
                                # including calls following URL extraction.
                                if extraction_completed:
                                    tool_result = {
                                        "success": False,
                                        "message": (
                                            "Tool call skipped because "
                                            "URL extraction has completed."
                                        ),
                                    }

                                    messages.append(
                                        ToolMessage(
                                            tool_call_id=tool_call_id,
                                            content=json.dumps(tool_result),
                                        )
                                    )
                                    continue

                                chosen_tool = tools_map.get(tool_name)

                                if chosen_tool is None:
                                    tool_result = {
                                        "success": False,
                                        "message": f"Unknown tool: {tool_name}",
                                    }

                                    messages.append(
                                        ToolMessage(
                                            tool_call_id=tool_call_id,
                                            content=json.dumps(tool_result),
                                        )
                                    )
                                    continue

                                # Resolve the element name and type using the latest
                                # page state before executing an interaction.
                                element_name = ""
                                element_type = ""
                                interaction_description = ""

                                if tool_name == self._interact_tool:
                                    element_id = tool_args.get("element_id")
                                    action = tool_args.get("action", "")
                                    value = tool_args.get("value", "")

                                    elements = current_page_state.get("elements", [])

                                    matched_element = next(
                                        (
                                            element
                                            for element in elements
                                            if element.get("element_id") == element_id
                                        ),
                                        None,
                                    )

                                    if matched_element:
                                        element_name = matched_element.get("name", "")
                                        element_type = matched_element.get("type", "")
                                    else:
                                        element_name = f"Unknown element (element_id={element_id})"

                                    if action == "click":
                                        interaction_description = "Click"
                                    elif action == "fill":
                                        interaction_description = f"Enter text: {value}"
                                    elif action == "check":
                                        interaction_description = "Check"
                                    elif action == "select":
                                        interaction_description = f"Select option: {value}"
                                    else:
                                        interaction_description = str(action)

                                try:
                                    tool_result = await chosen_tool.ainvoke(
                                        tool_args
                                    )

                                    if tool_name == self._page_state_tool:
                                        # Save the latest page state so that
                                        # state IDs can be mapped to element names.
                                        current_page_state = tool_result

                                    elif tool_name == self._interact_tool:
                                        # Record the actual interaction outcome.
                                        navigation_steps.append(
                                            {
                                                "element": element_name,
                                                "type": element_type,
                                                "interaction": interaction_description,
                                                "success": bool(
                                                    tool_result.get("success", False)
                                                ),
                                            }
                                        )

                                    elif tool_name == self._extract_urls_tool:
                                        extracted_links = normalize_job_links(
                                            tool_result
                                        )

                                        for job_link in extracted_links:
                                            site_candidates[job_link.url] = job_link

                                        tool_result = extracted_links

                                        navigation_steps.append(
                                            {
                                                "element": "Vacancy listing",
                                                "type": "listing",
                                                "interaction": "Extract vacancy links",
                                                "success": bool(extracted_links),
                                            }
                                        )

                                        extraction_completed = True
                                        termination_reason = "extraction_complete"

                                    tool_response = {
                                        "success": True,
                                        "data": tool_result,
                                    }

                                except Exception as exc:
                                    print(
                                        f"[!] Tool invocation failed for "
                                        f"{target_url}, tool={tool_name}: {exc}"
                                    )

                                    # Record failed interactions.
                                    if tool_name == self._interact_tool:
                                        navigation_steps.append(
                                            {
                                                "element": element_name,
                                                "type": element_type,
                                                "interaction": interaction_description,
                                                "success": False,
                                            }
                                        )

                                    elif tool_name == self._extract_urls_tool:
                                        navigation_steps.append(
                                            {
                                                "element": "Vacancy listing",
                                                "type": "listing",
                                                "interaction": "Extract vacancy links",
                                                "success": False,
                                            }
                                        )

                                    tool_response = {
                                        "success": False,
                                        "message": (
                                            f"Tool invocation failed for "
                                            f"{tool_name}: {exc}"
                                        ),
                                    }

                                messages.append(
                                    ToolMessage(
                                        tool_call_id=tool_call_id,
                                        content=json.dumps(
                                            tool_response,
                                            default=_serialize_default,
                                        ),
                                    )
                                )

                            if extraction_completed:
                                break

                        site_links = list(site_candidates.values())

                        print(
                            f"\n[*] Finished processing {target_url}: "
                            f"{termination_reason}; "
                            f"found {len(site_links)} candidate links."
                        )

                        navigation_results.append(
                            WebsiteNavigationResult(
                                url=target_url,
                                job_links=site_links,
                                result_count=len(site_links),
                                termination_reason=termination_reason,
                                experience={
                                    "target_url": target_url,
                                    "steps": navigation_steps,
                                    "termination_reason": termination_reason,
                                    "result_count": len(site_links),
                                },
                            )
                        )

                    except Exception as exc:
                        termination_reason = "site_error"

                        print(
                            f"[!] Failed to process {target_url}: {exc}"
                        )

                        # Preserve a result even when processing this site fails.
                        navigation_results.append(
                            WebsiteNavigationResult(
                                url=target_url,
                                job_links=list(site_candidates.values()),
                                result_count=len(site_candidates),
                                termination_reason=termination_reason,
                                experience={
                                    "target_url": target_url,
                                    "steps": navigation_steps,
                                    "termination_reason": termination_reason,
                                    "result_count": len(site_candidates),
                                    "error": str(exc),
                                },
                            )
                        )

                    finally:
                        if context is not None:
                            try:
                                await context.close()

                            except Exception as exc:
                                print(
                                    f"[!] Failed to close context for "
                                    f"{target_url}: {exc}"
                                )

            finally:
                await browser.close()

        return navigation_results
