"""Development helper: capture screenshots of the running BrickFlow UI.

Uses Playwright with the locally installed Microsoft Edge (no browser download).
Usage (app must be running):
    python scripts/ui_screenshots.py --url http://localhost:8501 --out <dir> [--video NAME]
        [--process] [--width 1400 --height 1000]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


def wait_ready(page: Page, timeout: float = 30) -> None:
    page.wait_for_selector('[data-testid="stSidebar"]', timeout=timeout * 1000)
    end = time.time() + timeout
    while time.time() < end:  # wait until Streamlit stops reporting "running"
        running = page.locator('[data-testid="stStatusWidget"]').count()
        stale = page.locator('[data-stale="true"]').count()  # elements left from the previous run
        if not running and not stale:
            break
        time.sleep(0.3)
    time.sleep(1.0)


def nav(page: Page, label: str) -> None:
    page.locator('[data-testid="stSidebar"]').get_by_text(label, exact=False).first.click()
    wait_ready(page)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", help="Sample video name to select from the samples folder")
    ap.add_argument("--process", action="store_true", help="Start processing and capture progress/results")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--wait", type=float, default=900, help="Max seconds to wait for processing")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.url)
        wait_ready(page)
        page.screenshot(path=str(out / "1_home.png"), full_page=True)

        if args.video:
            page.get_by_role("tab", name="Videos on this computer").click()
            time.sleep(0.5)
            page.get_by_role("button", name="Use this video").click()
            wait_ready(page)
            page.screenshot(path=str(out / "1b_home_selected.png"), full_page=True)
            page.get_by_role("button", name="Continue to configuration").click()
            wait_ready(page, 60)
            page.screenshot(path=str(out / "2_configure.png"), full_page=True)

            if args.process:
                page.get_by_role("button", name="Start processing").click()
                time.sleep(12)
                page.screenshot(path=str(out / "3_process.png"), full_page=True)
                end = time.time() + args.wait
                while time.time() < end:
                    if page.get_by_role("button", name="View results").count():
                        break
                    time.sleep(2)
                page.screenshot(path=str(out / "3b_process_done.png"), full_page=True)
                page.get_by_role("button", name="View results").click()
                wait_ready(page, 60)
                page.screenshot(path=str(out / "4_results.png"), full_page=True)
        browser.close()
    print("saved to", out)


if __name__ == "__main__":
    main()
