"""Generic scraper for any app's reviews on the Google Play Store.

Uses the `google-play-scraper` package, which reads Google's public
review-listing endpoint (the one rendered on the app's Play Store page).
No authentication or API key is required. Pass any Play Store package
name via `--app-id` to scrape a different business than Revolut.
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
BATCH_SIZE = 200
REQUEST_DELAY_SECONDS = 1.0


def fetch_play_store_reviews(
    app_id: str,
    lang: str,
    country: str,
    target_count: int,
) -> pd.DataFrame:
    """Fetch up to `target_count` reviews for a Play Store app/locale.

    Args:
        app_id: Play Store package name (e.g. "com.revolut.revolut").
        lang: Two-letter language code for the review listing.
        country: Two-letter country code for the review listing.
        target_count: Maximum number of reviews to collect for this locale.

    Returns:
        A DataFrame with one row per review.
    """
    collected: list[dict] = []
    continuation_token = None

    while len(collected) < target_count:
        batch, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=min(BATCH_SIZE, target_count - len(collected)),
            continuation_token=continuation_token,
        )
        if not batch:
            break
        collected.extend(batch)
        logger.info(
            "lang=%s country=%s: collected %d/%d reviews",
            lang,
            country,
            len(collected),
            target_count,
        )
        time.sleep(REQUEST_DELAY_SECONDS)

    if not collected:
        return pd.DataFrame()

    df = pd.DataFrame(collected)
    df = df.rename(
        columns={
            "reviewId": "review_id",
            "score": "rating",
            "content": "text",
            "at": "date",
            "thumbsUpCount": "thumbs_up",
        }
    )
    df["source"] = "google_play"
    df["language"] = lang
    df["country"] = country
    return df[
        [
            "review_id",
            "source",
            "rating",
            "text",
            "date",
            "thumbs_up",
            "language",
            "country",
        ]
    ]


def main() -> None:
    """Scrape Play Store reviews for any app, for several locales, and save them as CSV."""
    parser = argparse.ArgumentParser(
        description="Scrape Google Play Store reviews for any app."
    )
    parser.add_argument(
        "--app-id",
        required=True,
        help="Play Store package name, e.g. com.revolut.revolut",
    )
    parser.add_argument(
        "--business",
        required=True,
        help="Business name, used to tag rows and name the output file",
    )
    parser.add_argument("--count-per-locale", type=int, default=8000)
    parser.add_argument(
        "--locales",
        nargs="+",
        default=[
            "fr:fr",
            "en:us",
            "es:es",
            "de:de",
            "it:it",
            "pt:pt",
            "nl:nl",
            "pl:pl",
            "ro:ro",
            "sv:se",
            "tr:tr",
            "cs:cz",
            "hu:hu",
            "da:dk",
            "nb:no",
        ],
        help="List of lang:country pairs, e.g. fr:fr en:us. Restricted to Latin-script "
        "languages to stay compatible with the planned CamemBERT/RoBERTa models.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to play_store_reviews_<business>.csv",
    )
    args = parser.parse_args()
    output_name = args.output or f"play_store_reviews_{args.business}.csv"

    frames: list[pd.DataFrame] = []
    for locale in args.locales:
        lang, country = locale.split(":")
        df = fetch_play_store_reviews(args.app_id, lang, country, args.count_per_locale)
        if df.empty:
            logger.warning("No reviews fetched for locale %s", locale)
            continue
        frames.append(df)

    if not frames:
        logger.error("No reviews were collected for any locale.")
        return

    result = pd.concat(frames, ignore_index=True).drop_duplicates(subset="review_id")
    result["business"] = args.business
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / output_name
    result.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved %d reviews to %s", len(result), output_path)


if __name__ == "__main__":
    main()
