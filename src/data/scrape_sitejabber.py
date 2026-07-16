"""Scraper for Revolut reviews on Sitejabber.

Sitejabber embeds full review data (rating, title, body, author, date) as
JSON-LD (schema.org Organization -> review[]) directly in the server-rendered
HTML, so no HTML class parsing is needed. robots.txt allows crawling
`/reviews/` for a generic user agent.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.sitejabber.com/reviews/revolut.com"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 15
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _extract_organization_block(html: str) -> dict | None:
    """Extract the Organization JSON-LD block (containing reviews) from a page."""
    blocks = re.findall(
        r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S
    )
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == "Organization":
            return data
    return None


def fetch_sitejabber_reviews(max_pages: int = 20) -> pd.DataFrame:
    """Fetch all available Revolut reviews from Sitejabber.

    Pagination has no natural end signal (pages beyond the last one repeat
    earlier reviews instead of returning an empty list), so this stops once
    the number of unique reviews collected reaches the site's own
    `aggregateRating.reviewCount`, or after `max_pages` as a safety cap.

    Args:
        max_pages: Safety cap on the number of pages to request.

    Returns:
        A DataFrame with one row per review.
    """
    collected: dict[str, dict] = {}
    expected_total: int | None = None

    for page in range(1, max_pages + 1):
        response = requests.get(
            BASE_URL,
            params={"page": page},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            logger.warning("page=%d: HTTP %d", page, response.status_code)
            break

        org = _extract_organization_block(response.text)
        if org is None:
            break
        if expected_total is None:
            expected_total = int(org.get("aggregateRating", {}).get("reviewCount", 0))

        for entry in org.get("review", []):
            review_id = entry.get("url", "").split("#")[-1]
            collected[review_id] = {
                "review_id": review_id,
                "rating": int(
                    float(entry.get("reviewRating", {}).get("ratingValue", 0))
                ),
                "title": entry.get("headline", ""),
                "text": entry.get("reviewBody", ""),
                "date": entry.get("datePublished", ""),
            }

        logger.info(
            "collected %d/%s reviews (page %d)", len(collected), expected_total, page
        )
        time.sleep(REQUEST_DELAY_SECONDS)

        if expected_total and len(collected) >= expected_total:
            break

    if not collected:
        return pd.DataFrame()

    df = pd.DataFrame(collected.values())
    df["source"] = "sitejabber"
    return df[["review_id", "source", "rating", "title", "text", "date"]]


def main() -> None:
    """Scrape all Revolut Sitejabber reviews and save them as CSV."""
    parser = argparse.ArgumentParser(description="Scrape Revolut Sitejabber reviews.")
    parser.add_argument("--output", type=str, default="sitejabber_reviews.csv")
    args = parser.parse_args()

    df = fetch_sitejabber_reviews()
    if df.empty:
        logger.error("No reviews were collected.")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / args.output
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %d reviews to %s", len(df), output_path)


if __name__ == "__main__":
    main()
