import asyncio
import json

from playwright.async_api import async_playwright

from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from tools import BrowserTools, JobLink


MAX_ITERATIONS = 15
NAVIGATION_TIMEOUT_MS = 30_000


def serialize_default(obj: object) -> object:
    """Serialize supported objects into JSON-compatible values."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()

    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def normalize_job_links(result: object) -> list[JobLink]:
    """Validate and normalize a list of job vacancy links."""
    if not isinstance(result, list):
        raise TypeError(
            f"Expected a list of JobLink objects, got {type(result).__name__}"
        )

    return [
        item if isinstance(item, JobLink) else JobLink.model_validate(item)
        for item in result
    ]


class BrowserNavigationAgent:
    """Navigate careers websites and extract candidate vacancy URLs."""

    def __init__(
        self,
        target_urls: list[str],
        keywords: str,
        location: str,
        language: str,
    ) -> None:
        self.target_urls = target_urls
        self.keywords = keywords
        self.location = location
        self.language = language

        self.system_message = f"""
            You are an autonomous browser agent tasked with discovering
            job vacancy URLs.

            Search criteria:
            - Keywords: "{self.keywords}"
            - Location: "{self.location}"
            - Language: "{self.language}"

            Use this loop: OBSERVE → DECIDE → ACT → VERIFY → repeat.
            Minimize unnecessary interactions and prioritize actions that
            help discover vacancy URLs.

            Rules:
            1. ALWAYS call `get_page_state` before acting.
            2. Remove consent forms, cookie banners, modals, and other
               blockers when necessary.
            3. Navigate only relevant careers and job pages. Ignore
               unrelated product, blog, documentation, and generic
               search pages.
            4. Prefer job-related filters, links to vacancies/current
               openings, and career-specific search inputs, in that order.
               - Only fill inputs identified as relevant to job searching.
               - Never fill cookie, privacy, newsletter, login, or generic
                 site-search inputs.
               - Apply selected filters or search criteria when necessary.
            5. After EVERY `interact` call, call `get_page_state` again.
               Do NOT call `interact` more than once in a row.
            6. Base actions on the current observed state. Avoid arbitrary
               clicks, repeated ineffective actions, and navigation loops.
            7. Use the search criteria to guide navigation and available
               filtering, but do NOT perform semantic vacancy matching.
            8. Once navigation and filtering are complete, call
               `extract_urls` to collect all candidate vacancy URLs and
               their visible link text.
        """

    async def run(
        self,
        model: str,
        model_provider: str,
        temperature: float = 1,
        headless: bool = False,
    ) -> list[JobLink]:
        """Navigate target websites and return discovered vacancy links."""

        llm = init_chat_model(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
        )

        # Deduplicate candidates by URL while preserving discovery order.
        candidates: dict[str, JobLink] = {}

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=headless)

            try:
                for target_url in self.target_urls:
                    context = None
                    termination_reason = "iteration_limit"

                    try:
                        context = await browser.new_context()
                        page = await context.new_page()

                        print(f"\n[*] Navigating to {target_url}...")

                        try:
                            await page.goto(
                                target_url,
                                wait_until="domcontentloaded",
                                timeout=NAVIGATION_TIMEOUT_MS,
                            )
                        except Exception as exc:
                            termination_reason = "navigation_error"
                            print(
                                f"[!] Page load error for "
                                f"{target_url}: {exc}"
                            )
                            continue

                        browser_tools = BrowserTools(
                            page,
                            llm,
                            self.keywords,
                        )

                        # This agent only has browser navigation and
                        # URL extraction tools. Vacancy matching is separate.
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
                        tools_map = {tool.name: tool for tool in tools}

                        messages = [
                            SystemMessage(content=self.system_message),
                            HumanMessage(
                                content=(
                                    f"Begin by observing the page state "
                                    f"at {target_url}."
                                )
                            ),
                        ]

                        extraction_completed = False

                        for iteration in range(MAX_ITERATIONS):
                            print(
                                f"\n--- Iteration "
                                f"{iteration + 1}/{MAX_ITERATIONS} ---"
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

                                # Pair any remaining calls with a result,
                                # even when extraction has completed.
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

                                try:
                                    tool_result = await chosen_tool.ainvoke(
                                        tool_args
                                    )

                                    if tool_name == "extract_urls":
                                        extracted_links = normalize_job_links(
                                            tool_result
                                        )

                                        for job_link in extracted_links:
                                            candidates[job_link.url] = job_link

                                        tool_result = extracted_links
                                        extraction_completed = True
                                        termination_reason = "extraction_complete"

                                    successful_tool_result = {
                                        "success": True,
                                        "data": tool_result,
                                    }

                                except Exception as exc:
                                    print(
                                        f"[!] Tool invocation failed for "
                                        f"{target_url}, tool={tool_name}: "
                                        f"{exc}"
                                    )

                                    successful_tool_result = {
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
                                            successful_tool_result,
                                            default=serialize_default,
                                        ),
                                    )
                                )

                            if extraction_completed:
                                break

                        print(
                            f"[*] Finished processing {target_url}: "
                            f"{termination_reason}"
                        )

                    except Exception as exc:
                        termination_reason = "site_error"
                        print(
                            f"[!] Failed to process {target_url}: {exc}"
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

        return list(candidates.values())
