"""Harmonize the raw scraped review sources, across all businesses, into one dataset.

Combines Play Store and App Store raw exports for every scraped business
(plus the one-off Sitejabber export, Revolut only) into one common schema,
cleans the text, and writes the result to `data/processed/reviews.csv`.

Each business contributes its own `play_store_reviews_<business>.csv` /
`app_store_reviews_<business>.csv` pair (see `scrape_play_store.py`/
`scrape_app_store.py`), discovered by glob. The original Revolut exports
predate the multi-business `business` column and have no suffix in their
filename (`play_store_reviews.csv`, `app_store_reviews.csv`) -- these are
tagged `business="revolut"`.

Design notes (see project memory for the full reasoning):
- Play Store's `country` field only reflects the scraping query parameter,
  not the reviewer's real location (Play only filters reviews by language),
  so it is dropped rather than kept as misleading metadata.
- Exact-duplicate detection on (text, rating) alone is too aggressive: most
  "duplicates" are distinct users independently writing common short
  reviews ("super", "ok", "top"). Only (source, text, rating, date) is
  treated as a true duplicate signal.
- Language detection (`langdetect`) is unreliable below ~20 characters
  (verified empirically), so it is only attempted above that threshold and
  is never used to drop rows -- uncertain cases are labelled "unknown"
  rather than risking a wrong label.
"""

import html
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_COLUMNS = [
    "id",
    "business",
    "source",
    "rating",
    "title",
    "text",
    "date",
    "language",
    "country",
]
LANGUAGE_DETECTION_MIN_LENGTH = 20


def clean_text(value: object) -> str:
    """Normalize a free-text field: unescape HTML, normalize unicode, collapse whitespace.

    Args:
        value: Raw text value (may be NaN/float for missing entries).

    Returns:
        The cleaned string, or an empty string if the input was missing.
    """
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_latin_script(text: str) -> bool:
    """Check whether a text is predominantly written in Latin-script characters.

    Used as a safety net to keep the corpus compatible with the planned
    Latin-script models (CamemBERT/RoBERTa), regardless of how a review
    ended up in the raw data (e.g. a non-Latin comment posted under a
    Latin-language storefront).

    Args:
        text: Cleaned review text.

    Returns:
        True if at least half of the alphabetic characters are Latin, or if
        the text has no alphabetic characters at all (e.g. emoji-only).
    """
    alphabetic_chars = [c for c in text if c.isalpha()]
    if not alphabetic_chars:
        return True
    latin_count = sum(1 for c in alphabetic_chars if "LATIN" in unicodedata.name(c, ""))
    return (latin_count / len(alphabetic_chars)) >= 0.5


def detect_language(text: str) -> str:
    """Best-effort language detection, skipped for short or ambiguous text.

    Args:
        text: Cleaned review text.

    Returns:
        An ISO 639-1 language code, or "unknown" if detection was skipped
        or failed.
    """
    if len(text) < LANGUAGE_DETECTION_MIN_LENGTH:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _business_from_filename(path: Path, prefix: str) -> str:
    """Derive the business tag from a raw export's filename.

    Args:
        path: Path to a `<prefix>.csv` or `<prefix>_<business>.csv` file.
        prefix: The filename prefix, e.g. "play_store_reviews".

    Returns:
        The business suffix, or "revolut" for the original untagged exports.
    """
    suffix = path.stem[len(prefix) :]
    return suffix[1:] if suffix.startswith("_") else "revolut"


def load_play_store() -> pd.DataFrame:
    """Load and harmonize every Play Store raw export (one per business).

    Returns:
        A DataFrame matching `OUTPUT_COLUMNS`.
    """
    frames = []
    for path in sorted(RAW_DATA_DIR.glob("play_store_reviews*.csv")):
        df = pd.read_csv(path)
        df["business"] = _business_from_filename(path, "play_store_reviews")
        df["title"] = ""
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["country"] = pd.NA  # query-param artifact, not a real geographic signal
        df["id"] = "google_play:" + df["business"] + ":" + df["review_id"].astype(str)
        frames.append(df[OUTPUT_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def load_app_store() -> pd.DataFrame:
    """Load and harmonize every App Store raw export (one per business).

    Returns:
        A DataFrame matching `OUTPUT_COLUMNS`.
    """
    frames = []
    for path in sorted(RAW_DATA_DIR.glob("app_store_reviews*.csv")):
        df = pd.read_csv(path)
        df["business"] = _business_from_filename(path, "app_store_reviews")
        df["date"] = pd.to_datetime(
            df["date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        df["language"] = pd.NA
        df["id"] = "app_store:" + df["business"] + ":" + df["review_id"].astype(str)
        frames.append(df[OUTPUT_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def load_sitejabber() -> pd.DataFrame:
    """Load and harmonize the Sitejabber raw export (Revolut only).

    Returns:
        A DataFrame matching `OUTPUT_COLUMNS`.
    """
    df = pd.read_csv(RAW_DATA_DIR / "sitejabber_reviews.csv")
    df["business"] = "revolut"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["language"] = pd.NA
    df["country"] = pd.NA
    df["id"] = "sitejabber:" + df["review_id"].astype(str)
    return df[OUTPUT_COLUMNS]


def build_dataset() -> pd.DataFrame:
    """Load, harmonize, clean and deduplicate all raw review sources.

    Returns:
        The final harmonized DataFrame, ready for EDA.
    """
    raw = pd.concat(
        [load_play_store(), load_app_store(), load_sitejabber()], ignore_index=True
    )
    logger.info(
        "Loaded %d raw rows across %d businesses",
        len(raw),
        raw["business"].nunique(),
    )

    raw["title"] = raw["title"].apply(clean_text)
    raw["text"] = raw["text"].apply(clean_text)

    before = len(raw)
    raw = raw[raw["text"].str.len() > 0].copy()
    logger.info("Dropped %d rows with empty text", before - len(raw))

    before = len(raw)
    raw = raw[raw["rating"].isin([1, 2, 3, 4, 5])].copy()
    logger.info("Dropped %d rows with an invalid rating", before - len(raw))

    before = len(raw)
    raw = raw.drop_duplicates(
        subset=["business", "source", "text", "rating", "date"]
    ).copy()
    logger.info(
        "Dropped %d true duplicate rows (same business/source/text/rating/date)",
        before - len(raw),
    )

    before = len(raw)
    raw = raw[raw["text"].apply(is_latin_script)].copy()
    logger.info(
        "Dropped %d non-Latin-script rows (corpus is scoped to Latin-script "
        "languages for CamemBERT/RoBERTa compatibility)",
        before - len(raw),
    )

    needs_detection = raw["language"].isna()
    logger.info(
        "Running language detection on %d rows (App Store + Sitejabber)",
        needs_detection.sum(),
    )
    raw.loc[needs_detection, "language"] = raw.loc[needs_detection, "text"].apply(
        detect_language
    )

    raw = raw.sort_values("date").reset_index(drop=True)
    return raw[OUTPUT_COLUMNS]


def main() -> None:
    """Build the harmonized dataset and save it to data/processed/."""
    dataset = build_dataset()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "reviews.csv"
    dataset.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("Saved %d harmonized reviews to %s", len(dataset), output_path)
    logger.info("Rows per business:\n%s", dataset["business"].value_counts())
    logger.info("Rows per source:\n%s", dataset["source"].value_counts())
    logger.info(
        "Rows per language (top 10):\n%s", dataset["language"].value_counts().head(10)
    )
    logger.info("Rows per rating:\n%s", dataset["rating"].value_counts().sort_index())


if __name__ == "__main__":
    main()
