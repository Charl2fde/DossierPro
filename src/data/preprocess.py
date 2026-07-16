"""NLP preprocessing for the harmonized multi-business reviews dataset.

Produces a "classical" cleaned text variant suited for TF-IDF / classical ML
models (lowercased, lemmatized, stopwords removed, emojis converted to
words). The original `text` column is left untouched, since Transformer
models (CamemBERT/RoBERTa) work best on minimally-altered text and would
lose useful signal from stopword removal or lemmatization.

Performance note: simplemma's default dictionary cache only holds 8
languages in memory (LRU), and the corpus uses 15. Without enlarging it,
the dictionary is reloaded from disk on almost every call as languages
cycle in and out of the cache (verified by profiling: ~270s for 3k rows
vs ~5s after the fix). A custom `DefaultDictionaryFactory` with a larger
cache fixes this.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

import emoji
import pandas as pd
import stopwordsiso as siso
from simplemma.lemmatizer import Lemmatizer
from simplemma.strategies.default import DefaultStrategy
from simplemma.strategies.dictionaries.dictionary_factory import (
    DefaultDictionaryFactory,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "reviews.csv"
OUTPUT_PATH = PROCESSED_DATA_DIR / "reviews_nlp.csv"

# Modeling target: a 3-class sentiment label is more robust and generalizes
# better across businesses/sectors than the raw 1-5 rating, which is heavily
# imbalanced (81% 5-star) and left two classes (2, 3 stars) unlearnable in
# practice (see notebooks/03_modeling.ipynb).
RATING_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

# stopwordsiso uses "no" for Norwegian, simplemma uses "nb" -- only this one differs.
STOPWORDSISO_LANG_OVERRIDES = {"nb": "no"}
MIN_TOKEN_LENGTH = 2

# stopwordsiso's generic lists were built for SEO/text-mining and wrongly classify
# common polarity adjectives as stopwords (verified empirically: en flags "good",
# "great", "best"; fr flags "bien", "bon"; es flags "bueno", "buena"; pt flags "bom",
# "boa"; it flags "buono"; de flags "gut", "schlecht"; hu flags "jó"; sv flags "bra").
# These are exactly the words that carry sentiment signal for this task, so they are
# protected from removal regardless of what the underlying stopword list says.
SENTIMENT_PROTECTED_WORDS: dict[str, frozenset[str]] = {
    "fr": frozenset(
        {
            "bien",
            "bon",
            "bonne",
            "mauvais",
            "mauvaise",
            "nul",
            "nulle",
            "génial",
            "excellent",
        }
    ),
    "en": frozenset({"good", "great", "best", "bad", "worst", "love", "hate"}),
    "es": frozenset(
        {
            "bien",
            "bueno",
            "buena",
            "buenas",
            "buenos",
            "malo",
            "mala",
            "pésimo",
            "fantástico",
        }
    ),
    "pt": frozenset({"bom", "boa", "boas", "bons", "ótimo", "péssimo", "ruim"}),
    "it": frozenset({"buono", "buona", "ottimo", "ottima", "pessimo"}),
    "de": frozenset({"gut", "schlecht", "super", "toll", "schlimm"}),
    "nl": frozenset({"goed", "slecht", "geweldig"}),
    "pl": frozenset({"dobry", "dobra", "zły", "świetny"}),
    "ro": frozenset({"bun", "bună", "rău", "excelent"}),
    "sv": frozenset({"bra", "dålig", "perfekt"}),
    "tr": frozenset({"iyi", "kötü", "mükemmel"}),
    "cs": frozenset({"dobrý", "dobré", "špatný"}),
    "hu": frozenset({"jó", "rossz", "szuper"}),
    "da": frozenset({"god", "dårlig"}),
    "nb": frozenset({"god", "dårlig"}),
}

_lemmatizer = Lemmatizer(
    lemmatization_strategy=DefaultStrategy(
        dictionary_factory=DefaultDictionaryFactory(cache_max_size=20)
    )
)


@lru_cache(maxsize=1_000_000)
def _lemmatize_cached(token: str, lang: str) -> str:
    """Lemmatize a single token, cached across the whole run.

    Args:
        token: Lowercased token.
        lang: ISO 639-1 language code.

    Returns:
        The lemma, or the original token if lemmatization fails.
    """
    try:
        return _lemmatizer.lemmatize(token, lang)
    except Exception:
        return token


def _build_stopwords_by_language(languages: list[str]) -> dict[str, frozenset[str]]:
    """Precompute the stopword set for each language present in the dataset.

    Args:
        languages: Distinct language codes found in the dataset.

    Returns:
        A mapping from language code to its frozen stopword set (empty for
        "unknown" or unsupported languages).
    """
    stopwords_by_lang = {}
    for lang in languages:
        siso_code = STOPWORDSISO_LANG_OVERRIDES.get(lang, lang)
        if lang != "unknown" and siso.has_lang(siso_code):
            base_stopwords = frozenset(siso.stopwords(siso_code))
        else:
            base_stopwords = frozenset()
        stopwords_by_lang[lang] = base_stopwords - SENTIMENT_PROTECTED_WORDS.get(
            lang, frozenset()
        )
    return stopwords_by_lang


def clean_for_classical_ml(
    text: str, lang: str, stopwords_by_lang: dict[str, frozenset[str]]
) -> str:
    """Clean and lemmatize a review for TF-IDF / classical ML models.

    Args:
        text: Already-harmonized review text (see `build_dataset.py`).
        lang: ISO 639-1 language code, or "unknown".
        stopwords_by_lang: Precomputed stopword sets, keyed by language.

    Returns:
        Space-joined lowercased lemmas, with stopwords and 1-character
        tokens removed.
    """
    text = emoji.demojize(text, language="en")
    text = text.lower()
    tokens = re.findall(r"[\w'-]+", text, re.UNICODE)

    stopwords = stopwords_by_lang.get(lang, frozenset())
    kept_tokens = []
    for token in tokens:
        if len(token) < MIN_TOKEN_LENGTH or token in stopwords:
            continue
        if lang != "unknown":
            token = _lemmatize_cached(token, lang)
        kept_tokens.append(token)
    return " ".join(kept_tokens)


def build_nlp_dataset() -> pd.DataFrame:
    """Load the harmonized dataset and add the classical-ML text column.

    Returns:
        The dataset with an added `text_classical` column.
    """
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), INPUT_PATH)

    df["sentiment"] = df["rating"].map(RATING_TO_SENTIMENT)
    logger.info(
        "sentiment distribution: %s",
        df["sentiment"].value_counts(normalize=True).round(3).to_dict(),
    )

    stopwords_by_lang = _build_stopwords_by_language(df["language"].unique().tolist())

    # Sort by language so simplemma's dictionary cache stays warm instead of
    # thrashing between languages row by row (see module docstring).
    df = df.sort_values("language").reset_index(drop=True)
    df["text_classical"] = df.apply(
        lambda row: clean_for_classical_ml(
            row["text"], row["language"], stopwords_by_lang
        ),
        axis=1,
    )

    empty_after_cleaning = (df["text_classical"].str.len() == 0).sum()
    logger.info(
        "%d rows became empty after stopword/short-token removal (e.g. reviews that were only "
        "stopwords or a single short word) -- kept as-is, to be handled at the modeling stage",
        empty_after_cleaning,
    )

    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    """Build the NLP-preprocessed dataset and save it to data/processed/."""
    df = build_nlp_dataset()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Saved %d rows to %s", len(df), OUTPUT_PATH)


if __name__ == "__main__":
    main()
