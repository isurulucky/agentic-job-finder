from tools import BrowserTools, JobLink
from nav_result import WebsiteNavigationResult, normalize_job_links
from playwright.async_api import async_playwright
import re
import unicodedata

def normalize_element_name(name: str) -> str:
    """Normalize element names for reliable comparison."""
    name = name.casefold()

    # Replace Unicode punctuation with spaces.
    name = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in name
    )

    # Collapse consecutive whitespace and trim.
    return re.sub(r"\s+", " ", name).strip()

class ExperienceBasedNavigator:
    """Navigate careers websites deterministically and extract candidate vacancy URLs based on previous experience."""
    def __init__(
        self,
        navigation_timeout_millis: int = 30_000

    ) -> None:
        self.navigation_timeout_millis = navigation_timeout_millis

    async def run(
        self,
        target_urls: list[str],
        experiences: dict[str, list[dict]],
        headless: bool = False,
    ) -> list[WebsiteNavigationResult]:

        """Navigate target websites by deterministically replaying stored experiences."""

        navigation_results: list[WebsiteNavigationResult] = []

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=headless)

            try:
                for target_url in target_urls:

                    stored_experience = (experiences or {}).get(target_url)

                    if not stored_experience:
                        navigation_results.append(
                            WebsiteNavigationResult(
                                url=target_url,
                                termination_reason="no_experience",
                                experience={
                                    "target_url": target_url,
                                    "steps": [],
                                    "termination_reason": "no_experience",
                                    "result_count": 0,
                                },
                            )
                        )
                        continue

                    # Support both a single experience and a list of experiences.
                    if isinstance(stored_experience, dict):
                        stored_experience = [stored_experience]

                    print(
                        f"\n[*] Found {len(stored_experience)} "
                        f"experiences for {target_url}."
                    )

                    final_result = None

                    # Try experiences in order until one succeeds.
                    for experience_index, experience in enumerate(stored_experience):

                        print(f"Previous nav experience #1: {stored_experience[0]}")

                        print(
                            f"\n[*] Replaying experience "
                            f"{experience_index + 1}/{len(stored_experience)}"
                        )

                        context = None
                        site_candidates: dict[str, JobLink] = {}
                        navigation_steps: list[dict] = []

                        termination_reason = "replay_failed"
                        failure_message = None
                        extraction_completed = False

                        try:
                            # Start from scratch for every experience attempt.
                            context = await browser.new_context()
                            page = await context.new_page()

                            await page.goto(
                                target_url,
                                wait_until="domcontentloaded",
                                timeout=self.navigation_timeout_millis,
                            )

                            await page.wait_for_timeout(3000)

                            browser_tools = BrowserTools(page)

                            # Always obtain the initial page state.
                            current_page_state = (
                                await browser_tools.get_page_state()
                            )

                            for step in experience.get("steps", []):

                                # Replay only previously successful interactions.
                                if not step.get("success", False):
                                    continue

                                element_name = step.get("element", "")
                                stored_type = step.get("type", "")
                                interaction = step.get("interaction", "").strip()

                                # Handle vacancy URL extraction.
                                if (
                                    interaction.lower() == "extract vacancy links"
                                    or stored_type == "listing"
                                ):

                                    extracted_links = normalize_job_links(
                                        await browser_tools.extract_urls()
                                    )

                                    if not extracted_links:
                                        failure_message = (
                                            "URL extraction produced no results."
                                        )
                                        termination_reason = "extraction_failed"
                                        break

                                    for job_link in extracted_links:
                                        site_candidates[job_link.url] = job_link

                                    navigation_steps.append({
                                        "element": element_name,
                                        "type": stored_type,
                                        "interaction": interaction,
                                        "success": True,
                                    })

                                    extraction_completed = True
                                    termination_reason = "extraction_complete"
                                    break

                                # Find elements with the recorded name.
                                elements = current_page_state.get("elements", [])

                                stored_name = normalize_element_name(element_name)

                                # Normalize live element names once.
                                normalized_elements = [
                                    (
                                        element,
                                        normalize_element_name(
                                            element.get("name", "")
                                        ),
                                    )
                                    for element in elements
                                ]

                                # First, try exact name and type matching.
                                matching_elements = [
                                    element
                                    for element, live_name in normalized_elements
                                    if (
                                        live_name == stored_name
                                        and element.get("type") == stored_type
                                    )
                                ]

                                # If no exact match exists, try partial name matching.
                                if not matching_elements and stored_name:
                                    matching_elements = [
                                        element
                                        for element, live_name in normalized_elements
                                        if (
                                            element.get("type") == stored_type
                                            and live_name
                                            and (
                                                stored_name in live_name
                                                or live_name in stored_name
                                            )
                                        )
                                    ]

                                # Only consider enabled elements.
                                matching_elements = [
                                    element
                                    for element in matching_elements
                                    if not element.get("disabled", False)
                                ]

                                # Ensure at least one matching element exists.
                                if not matching_elements:
                                    failure_message = (
                                        f"Element not found: {element_name}"
                                    )
                                    termination_reason = "element_not_found"
                                    break

                                # Convert the recorded interaction into a tool action.
                                value = ""

                                if interaction == "Click":
                                    action = "click"

                                elif interaction.startswith("Enter text: "):
                                    action = "fill"
                                    value = interaction[len("Enter text: "):]

                                elif interaction == "Check":
                                    action = "check"

                                elif interaction.startswith("Select option: "):
                                    action = "select"
                                    value = interaction[len("Select option: "):]

                                else:
                                    failure_message = (
                                        f"Unsupported interaction: {interaction}"
                                    )
                                    termination_reason = "unsupported_interaction"
                                    break

                                interaction_succeeded = False

                                # Try each matching element until one succeeds.
                                for candidate_index, candidate in enumerate(
                                    matching_elements
                                ):

                                    # Use the latest page state and resolve the
                                    # candidate again after any previous failure.
                                    if candidate_index > 0:
                                        elements = current_page_state.get(
                                            "elements", []
                                        )

                                        candidate_name = normalize_element_name(
                                            candidate.get("name", "")
                                        )

                                        refreshed_candidates = [
                                            element
                                            for element in elements
                                            if (
                                                normalize_element_name(
                                                    element.get("name", "")
                                                ) == candidate_name
                                                and element.get("type") == stored_type
                                                and not element.get(
                                                    "disabled", False
                                                )
                                                and element.get("visible", True)
                                            )
                                        ]

                                        if not refreshed_candidates:
                                            continue

                                        candidate = refreshed_candidates[0]

                                    if not candidate.get("visible", True):
                                        continue

                                    try:
                                        tool_result = await browser_tools.interact(
                                            state_version=current_page_state[
                                                "state_version"
                                            ],
                                            element_id=candidate["element_id"],
                                            action=action,
                                            value=value,
                                        )

                                        if tool_result.get("success", False):
                                            interaction_succeeded = True

                                            navigation_steps.append({
                                                "element": element_name,
                                                "type": candidate.get("type", ""),
                                                "interaction": interaction,
                                                "success": True,
                                            })

                                            # Refresh page state after success.
                                            current_page_state = (
                                                await browser_tools.get_page_state()
                                            )

                                            break

                                        print(
                                            f"\n[!] Interaction failed for "
                                            f"'{candidate.get('name', '')}': "
                                            f"{tool_result.get('message', 'Unknown error')}"
                                        )

                                        # Refresh page state before trying
                                        # another candidate.
                                        current_page_state = (
                                            await browser_tools.get_page_state()
                                        )

                                    except Exception as exc:
                                        error_message = str(exc)

                                        if "Execution context was destroyed" in error_message:
                                            print(
                                                f"\n[*] Navigation detected after interacting with "
                                                f"'{candidate.get('name', '')}'."
                                            )

                                            try:
                                                await page.wait_for_load_state(
                                                    "domcontentloaded",
                                                    timeout=10000,
                                                )
                                            except Exception as e:
                                                print(f"[!] Page re-load failed in after context destroyed: {e}")

                                            current_page_state = await browser_tools.get_page_state()

                                            # Treat the interaction as successful if navigation occurred.
                                            interaction_succeeded = True

                                            navigation_steps.append({
                                                "element": element_name,
                                                "type": candidate.get("type", ""),
                                                "interaction": interaction,
                                                "success": True,
                                            })

                                            break
                                        print(
                                            f"\n[!] Interaction failed for "
                                            f"'{candidate.get('name', '')}': {exc}"
                                        )

                                        # Refresh page state before retrying.
                                        current_page_state = (
                                            await browser_tools.get_page_state()
                                        )

                                if not interaction_succeeded:
                                    failure_message = (
                                        f"Interaction failed for all matching "
                                        f"elements: {element_name}"
                                    )
                                    termination_reason = "interaction_failed"
                                    break

                            # An experience must include successful URL extraction.
                            if not extraction_completed and not failure_message:
                                failure_message = (
                                    "Experience completed without URL extraction."
                                )
                                termination_reason = "extraction_not_attempted"

                        except Exception as exc:
                            failure_message = str(exc)
                            termination_reason = "replay_error"

                        finally:
                            if context is not None:
                                await context.close()

                        if failure_message:
                            print(
                                f"\n[!] Experience replay failed: {failure_message}"
                            )

                        final_result = WebsiteNavigationResult(
                            url=target_url,
                            job_links=list(site_candidates.values()),
                            result_count=len(site_candidates),
                            termination_reason=termination_reason,
                            experience={
                                "target_url": target_url,
                                "steps": navigation_steps,
                                "termination_reason": termination_reason,
                                "result_count": len(site_candidates),
                                **(
                                    {"error": failure_message}
                                    if failure_message
                                    else {}
                                ),
                            },
                        )

                        if extraction_completed:
                            print(
                                f"\n[*] Deterministic navigation successful. "
                                f"Found {len(site_candidates)} vacancy URLs."
                            )
                            break

                    if final_result is not None:
                        navigation_results.append(final_result)

            finally:
                await browser.close()

        return navigation_results
