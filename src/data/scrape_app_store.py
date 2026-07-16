"""Generic scraper for any app's reviews on the Apple App Store.

Uses Apple's public customer-reviews RSS feed (JSON format). This endpoint
is publicly exposed by Apple for display purposes, requires no API key,
and exposes up to ~500 of the most recent reviews per country (10 pages
of 50 reviews). Pass any numeric App Store track ID via `--app-id` to
scrape a different business than Revolut.
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MAX_PAGES = 10
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 15


def _feed_url(country: str, page: int, app_id: str) -> str:
    """Build the RSS feed URL for a given country and page number."""
    return (
        f"https://itunes.apple.com/{country}/rss/customerreviews/"
        f"page={page}/id={app_id}/sortby=mostrecent/json"
    )


def fetch_app_store_reviews(app_id: str, country: str) -> pd.DataFrame:
    """Fetch all available reviews (up to ~500) for an App Store app/country.

    Args:
        app_id: Numeric App Store track ID (e.g. "932493382").
        country: Two-letter App Store storefront country code.

    Returns:
        A DataFrame with one row per review.
    """
    collected: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        response = requests.get(
            _feed_url(country, page, app_id), timeout=REQUEST_TIMEOUT_SECONDS
        )
        if response.status_code != 200:
            logger.warning(
                "country=%s page=%d: HTTP %d", country, page, response.status_code
            )
            break

        entries = response.json().get("feed", {}).get("entry", [])
        # The first entry on page 1 is sometimes app metadata, not a review.
        entries = [e for e in entries if "im:rating" in e]
        if not entries:
            break

        for entry in entries:
            collected.append(
                {
                    "review_id": entry["id"]["label"],
                    "rating": int(entry["im:rating"]["label"]),
                    "text": entry.get("content", {}).get("label", ""),
                    "title": entry.get("title", {}).get("label", ""),
                    "date": entry["updated"]["label"],
                }
            )

        logger.info(
            "country=%s: collected %d reviews (page %d)", country, len(collected), page
        )
        time.sleep(REQUEST_DELAY_SECONDS)

    if not collected:
        return pd.DataFrame()

    df = pd.DataFrame(collected)
    df["source"] = "app_store"
    df["country"] = country
    return df[["review_id", "source", "rating", "text", "title", "date", "country"]]


def main() -> None:
    """Scrape App Store reviews for any app, for several countries, and save them as CSV."""
    parser = argparse.ArgumentParser(
        description="Scrape Apple App Store reviews for any app."
    )
    parser.add_argument(
        "--app-id", required=True, help="Numeric App Store track ID, e.g. 932493382"
    )
    parser.add_argument(
        "--business",
        required=True,
        help="Business name, used to tag rows and name the output file",
    )
    parser.add_argument("--countries", nargs="+", default=["fr", "us", "gb"])
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to app_store_reviews_<business>.csv",
    )
    args = parser.parse_args()
    output_name = args.output or f"app_store_reviews_{args.business}.csv"

    frames: list[pd.DataFrame] = []
    for country in args.countries:
        df = fetch_app_store_reviews(args.app_id, country)
        if df.empty:
            logger.warning("No reviews fetched for country %s", country)
            continue
        frames.append(df)

    if not frames:
        logger.error("No reviews were collected for any country.")
        return

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="review_id")
    result["business"] = args.business
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / output_name
    result.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %d reviews to %s", len(result), output_path)


if __name__ == "__main__":
    main()
