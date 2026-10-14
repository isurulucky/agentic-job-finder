import asyncio
import json

from playwright.async_api import async_playwright

from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from pydantic import BaseModel

from tools import BrowserTools, JobLink, list_matching_vacancies


def serialize_default(obj: object) -> object:
    """Serialize supported objects into JSON-compatible values."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def normalize_job_links(result: object) -> list[JobLink]:
    """Validate and normalize a list of job vacancy links."""
    if not isinstance(result, list):
        raise TypeError(f"Expected a list of JobLink objects, got {type(result).__name__}")

    return [
        item if isinstance(item, JobLink) else JobLink.model_validate(item) for item in result
    ]


class AgentLoop:

    def __init__(self, target_urls: list[str], keywords: str, location: str, language: str) -> None:
        self.target_urls = target_urls

        self.keywords = keywords
        self.location = location
        self.language = language

        self.system_message = f"""
            You are an autonomous web agent finding job vacancies matching:

            - Keywords: "{self.keywords}"
            - Location: "{self.location}"
            - Language: "{self.language}"

            Use this loop: OBSERVE → DECIDE → ACT → VERIFY → repeat. Use as minimum loop iterations as possible. At every step, choose the single next action that makes the most progress toward finding matching vacancies.

            Rules:
            1. ALWAYS call `get_page_state` before acting.
            2. Remove consent forms, cookie banners, modals, and other blockers in each step.
            3. Only navigate career/job pages. Ignore generic sites, product, blog, documentation, or unrelated search pages.
            4. Prefer, in order:
                a) job opening/career related filters, such as drop downs, check boxes, radio buttons, buttons, etc.
                b) links to job-search/vacancies/current openings pages,
                c) career-specific search inputs.
                - Only enter search text on searchable text fields, not anything else. Never fill cookie, privacy, newsletter, login, or generic site-search inputs.
                - Only fill inputs whose label, placeholder, name, or surrounding context.
               If necessary, click the relevant button to filter by the selected drop down items, ticked boxes and/or checkboxes, search criteria, etc.
            5. After EVERY `interact`, call `get_page_state` again. Do NOT call `interact` more than once in a row.
            6. Base each action on the current observed state. Avoid arbitrary clicks, repeated ineffective actions, and navigation loops.
            7. Treat keywords as semantic matches against job titles/descriptions and location as the actual vacancy location. Do not invent matches.
            8. Once filtering/searching is complete, call `extract_urls`.
            9. With the results returned from `extract_urls`, call `list_matching_vacancies` and terminate.
        """

    async def run(self, model: str, model_provider: str, temperature: float = 1, headless: bool = False) -> dict[str, str]:
        llm = init_chat_model(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
        )

        results: dict[str, str] = {}

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=headless)

            try:
                for target_url in self.target_urls:
                    context = None
                    termination_reason = "iteration_limit"

                    try:
                        # Create an isolated browser context for each website.
                        context = await browser.new_context()
                        page = await context.new_page()

                        print(f"\n[*] Navigating to {target_url}...")

                        try:
                            await page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
                        except Exception as e:
                            print(f"[!] Page load error for {target_url}: {e}")
                            termination_reason = "navigation_error"
                            continue

                        # Instantiate BrowserTools.
                        browser_tools = BrowserTools(page, llm, self.keywords)

                        async def list_matching_vacancies_tool(extracted_urls: list[JobLink]) -> list[JobLink]:
                            """Identify vacancy links matching the specified keywords.

                            Args:
                                extracted_urls: Candidate vacancy URLs and
                                    their visible link text.

                            Returns:
                                A list of matching JobLink instances.
                                Returns an empty list when no vacancies match.
                            """
                            return await list_matching_vacancies(
                                jobLinks=extracted_urls,
                                llm=llm,
                                keywords=self.keywords,
                            )

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
                            StructuredTool.from_function(
                                coroutine=list_matching_vacancies_tool,
                                name="list_matching_vacancies_tool",
                            ),
                        ]

                        llm_with_tools = llm.bind_tools(tools)
                        tools_map = {t.name: t for t in tools}

                        messages = [
                            SystemMessage(content=self.system_message),
                            HumanMessage(content="Begin by observing the page state."),
                        ]

                        iterations = 15

                        for iteration in range(iterations):
                            print(f"\n--- Iteration {iteration + 1}/{iterations} ---")

                            # Ask the LLM to select the next tool.
                            try:
                                ai_msg = await llm_with_tools.ainvoke(messages)
                            except Exception as e:
                                print(f"[!] Model invocation failed for {target_url}: {e}")
                                termination_reason = "model_error"
                                break

                            messages.append(ai_msg)

                            if not ai_msg.tool_calls:
                                print(f"[*] Agent stopped calling tools. Text response: {ai_msg.content}")
                                termination_reason = "agent_stopped"
                                break

                            matching_completed = False
                            browser_unavailable = False

                            # Execute the selected tool calls sequentially.
                            for tool_call in ai_msg.tool_calls:
                                tool_call_id = tool_call["id"]
                                tool_name = tool_call["name"]
                                tool_args = tool_call.get("args", {})

                                if matching_completed:
                                    tool_result = {
                                        "success": False,
                                        "message": (
                                            "Tool call skipped because "
                                            "vacancy matching has completed."
                                        ),
                                    }
                                    messages.append(
                                        ToolMessage(
                                            tool_call_id=tool_call_id,
                                            content=json.dumps(
                                                tool_result,
                                                default=serialize_default,
                                            ),
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
                                            content=json.dumps(
                                                tool_result,
                                                default=serialize_default,
                                            ),
                                        )
                                    )
                                    continue

                                try:
                                    tool_result = await chosen_tool.ainvoke(
                                        tool_args
                                    )

                                    if (tool_name == "list_matching_vacancies_tool"):
                                        tool_result = normalize_job_links(tool_result)

                                        for result in tool_result:
                                            results[result.url] = result.text

                                        matching_completed = True
                                        termination_reason = "matching_complete"

                                except Exception as e:
                                    print( f"[!] Tool invocation failed for {target_url} with tool args {tool_args}: {e}")
                                    tool_result = {
                                        "success": False,
                                        "message": (
                                            f"Tool invocation failed for "
                                            f"{tool_name}: {e}"
                                        ),
                                    }

                                    if page.is_closed():
                                        browser_unavailable = True
                                        termination_reason = "browser_unavailable"

                                else:
                                    tool_result = {
                                        "success": True,
                                        "data": tool_result,
                                    }

                                messages.append(
                                    ToolMessage(
                                        tool_call_id=tool_call_id,
                                        content=json.dumps(
                                            tool_result,
                                            default=serialize_default,
                                        ),
                                    )
                                )

                                if browser_unavailable:
                                    break

                            if matching_completed or browser_unavailable:
                                break

                        print(f"\n[*] Finished processing {target_url}: {termination_reason}")

                    except Exception as e:
                        # Isolate unexpected failures to the current website.
                        termination_reason = "site_error"
                        print(f"[!] Failed to process {target_url}: {e}")

                    finally:
                        # Always release the per-site browser context.
                        if context is not None:
                            try:
                                await context.close()
                            except Exception as e:
                                print(f"[!] Failed to close context for {target_url}: {e}")

            finally:
                await browser.close()

        return results

# if __name__ == "__main__":
#     target_urls = [
#         "https://www.coolblue.nl/en/vacancies",
#         "https://catawiki.careers/vacancies",
#         "https://careers.adyen.com/",
#     ]

#     # target_url = "https://www.coolblue.nl/en/vacancies"
#     # target_url = "https://www.leaseweb.com/en/about-us/career"
#     # target_url = "https://catawiki.careers/vacancies"
#     # target_url = "https://careers.adyen.com/"
#     # target_url = "https://careers.bol.com/en/"
#     # target_url = "https://www.personio.com/about-personio/careers/"

#     keywords = "Software Engineer"
#     location = "Europe"
#     language = "English"

#     agent_loop = AgentLoop(
#         target_urls,
#         keywords,
#         location,
#         language,
#     )

#     model = "o4-mini"
#     provider = "openai"

#     vacancies = asyncio.run(
#         agent_loop.run(
#             model,
#             provider,
#         )
#     )

#     if len(vacancies) > 0:
#         print(f"Openings/vacancies found: {len(vacancies)}")

#     for url in vacancies:
#         print(url)
