"""Screenshot the running console with Playwright (for design review and the README).

  uv run streamlit run app.py --server.headless true --server.port 8599 &
  uv run python scripts/screenshot.py --port 8599 --out docs/img
"""
import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--tabs", nargs="*", default=["Triage", "Insights", "Audit trail"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 1600, "height": args.height}, device_scale_factor=1)
        page.goto(f"http://localhost:{args.port}", wait_until="networkidle")
        page.wait_for_selector("text=Change Guardian", timeout=60_000)
        time.sleep(6)  # let component iframes and charts render
        for name in args.tabs:
            if name != "Triage":
                for frame in page.frames:
                    btn = frame.get_by_role("tab", name=name)
                    if btn.count():
                        btn.first.click()
                        break
                time.sleep(6)
            path = args.out / f"{name.lower().replace(' ', '-')}.png"
            page.screenshot(path=str(path), full_page=True)
            print("saved", path)


if __name__ == "__main__":
    main()
