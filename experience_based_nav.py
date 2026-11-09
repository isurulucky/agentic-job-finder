from tools import BrowserTools, JobLink
from nav_result import WebsiteNavigationResult, normalize_job_links
from playwright.async_api import async_playwright

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

                    print(
                        f"\n[*] Found {len(stored_experience)} "
                        f"experiences for {target_url}."
                    )

                    final_result = None

                    # Try experiences in order until one succeeds.
                    for experience_index, experience in enumerate(
                        stored_experience
                    ):

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
                            current_page_state = await browser_tools.get_page_state()

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
                                elements = current_page_state.get(
                                    "elements", []
                                )

                                matching_elements = [
                                    element
                                    for element in elements
                                    if element.get("name") == element_name
                                ]

                                if not matching_elements:
                                    failure_message = (
                                        f"Element not found: {element_name}"
                                    )
                                    termination_reason = "element_not_found"
                                    break

                                # Cross-check the stored type against the live type.
                                matching_types = [
                                    element
                                    for element in matching_elements
                                    if element.get("type") == stored_type
                                ]

                                if not matching_types:
                                    live_types = [
                                        element.get("type")
                                        for element in matching_elements
                                    ]

                                    failure_message = (
                                        f"Element type mismatch for "
                                        f"'{element_name}'. "
                                        f"Stored: {stored_type}, "
                                        f"Live: {live_types}"
                                    )
                                    termination_reason = "element_type_mismatch"
                                    break

                                # Select an available element.
                                available_elements = [
                                    element
                                    for element in matching_types
                                    if not element.get("disabled", False)
                                    and element.get("visible", True)
                                ]

                                if not available_elements:
                                    failure_message = (
                                        f"Element unavailable: {element_name}"
                                    )
                                    termination_reason = "element_unavailable"
                                    break

                                element = available_elements[0]

                                # Convert the recorded interaction into a tool action.
                                value = ""

                                if interaction.lower() == "click":
                                    action = "click"

                                elif interaction.startswith("Enter text: "):
                                    action = "fill"
                                    value = interaction[len("Enter text: "):]

                                elif interaction.lower() == "check":
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

                                # Execute the interaction using the latest page state.
                                try:
                                    tool_result = await browser_tools.interact(
                                        state_version=current_page_state["state_version"],
                                        element_id=element["element_id"],
                                        action=action,
                                        value=value,
                                    )

                                except Exception as exc:
                                    failure_message = str(exc)
                                    termination_reason = "interaction_failed"
                                    break

                                if not tool_result.get("success", False):
                                    failure_message = tool_result.get(
                                        "message",
                                        f"Interaction failed: {interaction}",
                                    )
                                    termination_reason = "interaction_failed"
                                    break

                                navigation_steps.append({
                                    "element": element_name,
                                    "type": element.get("type", ""),
                                    "interaction": interaction,
                                    "success": True,
                                })

                                # Refresh page state after every interaction.
                                current_page_state = (
                                    await browser_tools.get_page_state()
                                )

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
                                f"[!] Experience replay failed: {failure_message}"
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
                                f"[*] Deterministic navigation successful. "
                                f"Found {len(site_candidates)} vacancy URLs."
                            )
                            break

                    if final_result is not None:
                        navigation_results.append(final_result)

            finally:
                await browser.close()

        return navigation_results
