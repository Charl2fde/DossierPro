"""Streamlit dashboard for the review sentiment analysis model.

Four views: a live prediction form against the FastAPI `/predict` endpoint,
a batch analysis tool against `/predict/batch` for a whole reviews dataset,
dataset-level statistics (from the harmonized multi-business dataset), and a
comparison of the two trained models.
"""

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "src" / "models" / "weights"
BATCH_CHUNK_SIZE = (
    500  # must match BatchReviewsRequest's max_length in src/api/schemas.py
)

st.set_page_config(page_title="Review Sentiment Dashboard", layout="wide")


@st.cache_data
def load_dataset_summary() -> pd.DataFrame:
    """Load only the columns needed for the statistics view, for speed."""
    return pd.read_csv(
        DATA_DIR / "reviews_nlp.csv",
        usecols=["business", "source", "rating", "sentiment", "language", "date"],
        low_memory=False,
    )


CLASSICAL_MODEL_LABELS = {
    "random_forest": "TF-IDF + Random Forest",
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "naive_bayes": "TF-IDF + Naive Bayes",
}


@st.cache_data
def load_model_metrics() -> dict:
    """Load the saved test-set metrics for every trained model (classical + transformer)."""
    metrics = {}
    for key, label in CLASSICAL_MODEL_LABELS.items():
        metrics_path = WEIGHTS_DIR / f"tfidf_{key}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics[label] = json.load(f)["test"]
    with open(WEIGHTS_DIR / "xlm_roberta" / "metrics.json", encoding="utf-8") as f:
        metrics["XLM-RoBERTa"] = json.load(f)["test"]
    return metrics


def call_predict_api(text: str) -> dict | None:
    """Call the /predict endpoint and return the parsed response, or None on error."""
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Impossible de contacter l'API ({API_URL}) : {exc}")
        return None


def call_predict_batch_api(texts: list[str]) -> list[dict] | None:
    """Call /predict/batch in chunks and return the concatenated predictions.

    Args:
        texts: All review texts to analyze (may exceed the API's per-call limit).

    Returns:
        One prediction dict per input text, in order, or None on error.
    """
    predictions: list[dict] = []
    chunks = [
        texts[i : i + BATCH_CHUNK_SIZE] for i in range(0, len(texts), BATCH_CHUNK_SIZE)
    ]
    progress = st.progress(0.0, text="Analyse en cours...")
    for i, chunk in enumerate(chunks):
        try:
            response = requests.post(
                f"{API_URL}/predict/batch", json={"reviews": chunk}, timeout=120
            )
            response.raise_for_status()
            predictions.extend(response.json()["predictions"])
        except requests.RequestException as exc:
            st.error(f"Impossible de contacter l'API ({API_URL}) : {exc}")
            progress.empty()
            return None
        progress.progress(
            (i + 1) / len(chunks), text=f"Analyse en cours... ({i + 1}/{len(chunks)})"
        )
    progress.empty()
    return predictions


st.title("Analyse intelligente des avis clients")
st.caption(
    "Outil generique de sentiment analysis -- compatible avec n'importe quelle boutique/business."
)

tab_predict, tab_batch, tab_stats, tab_models = st.tabs(
    [
        "Prediction en direct",
        "Analyse d'un dataset",
        "Statistiques du dataset",
        "Performance des modeles",
    ]
)

with tab_predict:
    st.subheader("Tester le modele sur un avis")
    st.caption(
        f"Appelle l'API FastAPI sur {API_URL} -- lancez-la avec `uvicorn src.api.main:app`."
    )

    def _apply_example() -> None:
        st.session_state["review_text"] = st.session_state["example_select"]

    st.selectbox(
        "Exemple rapide (optionnel)",
        [
            "",
            "Great app, very easy to use!",
            "Mon compte a ete bloque sans explication, catastrophique.",
            "App muy buena, la recomiendo",
        ],
        key="example_select",
        on_change=_apply_example,
    )
    text = st.text_area("Texte de l'avis", height=120, key="review_text")

    if (
        st.button("Predire le sentiment", type="primary", key="predict_button")
        and text.strip()
    ):
        with st.spinner("Prediction en cours..."):
            result = call_predict_api(text)
        if result:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Sentiment predit", result["sentiment"].capitalize())
                st.metric("Confiance", f"{result['confidence']:.1%}")
            with col2:
                probs = result["probabilities"]
                st.bar_chart(pd.Series(probs))
                dominant = max(probs, key=probs.get)
                st.caption(
                    f"Probabilite attribuee par le modele a chacun des 3 sentiments "
                    f"possibles. Ici, '{dominant}' domine avec {probs[dominant]:.1%} -- "
                    "plus une barre est haute, plus le modele est confiant dans cette "
                    "categorie."
                )

with tab_batch:
    st.subheader("Analyser un dataset d'avis complet")
    st.caption(
        "Importez un CSV d'avis (ex. l'export d'une entreprise) pour obtenir "
        "un score global et la repartition positif/negatif/neutre."
    )

    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"], key="batch_upload")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        text_column = st.selectbox(
            "Colonne contenant le texte de l'avis",
            batch_df.columns,
            key="batch_text_column",
        )

        if st.button("Analyser le dataset", type="primary", key="batch_analyze_button"):
            texts = batch_df[text_column].dropna().astype(str).tolist()
            texts = [t for t in texts if t.strip()]
            if not texts:
                st.warning("Aucun texte exploitable dans la colonne selectionnee.")
            else:
                predictions = call_predict_batch_api(texts)
                if predictions:
                    results_df = batch_df.loc[
                        batch_df[text_column].astype(str).str.strip() != ""
                    ].copy()
                    results_df = results_df.iloc[: len(predictions)].copy()
                    results_df["sentiment"] = [p["sentiment"] for p in predictions]
                    results_df["confidence"] = [p["confidence"] for p in predictions]

                    total = len(predictions)
                    positive_count = sum(
                        1 for p in predictions if p["sentiment"] == "positive"
                    )
                    negative_count = sum(
                        1 for p in predictions if p["sentiment"] == "negative"
                    )
                    neutral_count = total - positive_count - negative_count
                    overall_score = (positive_count - negative_count) / total

                    st.write("### Resultat global")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Avis analyses", f"{total:,}")
                    col2.metric("% positif", f"{positive_count / total:.1%}")
                    col3.metric("% negatif", f"{negative_count / total:.1%}")
                    col4.metric(
                        "Score global",
                        f"{overall_score:+.2f}",
                        help="De -1 (tout negatif) a +1 (tout positif)",
                    )

                    sentiment_counts = {
                        "negatif": negative_count,
                        "neutre": neutral_count,
                        "positif": positive_count,
                    }
                    st.bar_chart(pd.Series(sentiment_counts))
                    dominant_label = max(sentiment_counts, key=sentiment_counts.get)
                    st.caption(
                        f"Repartition des {total:,} avis par sentiment. La categorie "
                        f"'{dominant_label}' domine. Le score global ({overall_score:+.2f}) "
                        "resume cette repartition en une seule valeur : proche de +1 si "
                        "la majorite des avis sont positifs, proche de -1 si la majorite "
                        "sont negatifs."
                    )

                    st.write("### Detail par avis")
                    st.dataframe(results_df)
                    st.download_button(
                        "Telecharger les resultats (CSV)",
                        results_df.to_csv(index=False).encode("utf-8"),
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )

with tab_stats:
    st.subheader("Statistiques du dataset harmonise (multi-business)")
    df = load_dataset_summary()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total avis", f"{len(df):,}")
    col1.metric("Businesses", df["business"].nunique())
    col2.metric("Langues", df["language"].nunique())
    col2.metric("Note moyenne", f"{df['rating'].mean():.2f} / 5")
    col3.metric("% positif", f"{(df['sentiment'] == 'positive').mean():.1%}")
    col3.metric("% negatif", f"{(df['sentiment'] == 'negative').mean():.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribution des sentiments**")
        st.bar_chart(df["sentiment"].value_counts())
        sentiment_pct = df["sentiment"].value_counts(normalize=True)
        st.caption(
            f"{sentiment_pct.get('positive', 0):.1%} des avis sont positifs, "
            f"{sentiment_pct.get('negative', 0):.1%} negatifs et "
            f"{sentiment_pct.get('neutral', 0):.1%} neutres. Ce fort desequilibre "
            "(peu d'avis neutres) explique pourquoi les modeles ont plus de mal a "
            "reconnaitre cette categorie (voir l'onglet Performance)."
        )
    with col2:
        st.write("**Top 10 langues**")
        st.bar_chart(df["language"].value_counts().head(10))
        top_lang = df["language"].value_counts().idxmax()
        st.caption(
            f"Langues les plus representees dans le dataset ('{top_lang}' en tete). "
            "Le modele XLM-RoBERTa est multilingue et a ete entraine sur l'ensemble "
            "de ces langues."
        )

    st.write("**Avis par business**")
    st.bar_chart(df["business"].value_counts())
    top_business = df["business"].value_counts().idxmax()
    st.caption(
        f"Nombre d'avis collectes par entreprise ('{top_business}' en tete). "
        "Melanger plusieurs secteurs (banque, e-commerce, transport, hotellerie...) "
        "permet au modele de mieux generaliser a n'importe quelle nouvelle "
        "entreprise plutot que de se specialiser sur une seule."
    )

    st.write("**Note moyenne par business**")
    ratings_by_business = df.groupby("business")["rating"].mean()
    st.bar_chart(ratings_by_business)
    best_business = ratings_by_business.idxmax()
    worst_business = ratings_by_business.idxmin()
    st.caption(
        f"Note moyenne par entreprise (sur 5). '{best_business}' a la meilleure "
        f"moyenne, '{worst_business}' la plus basse. Ces ecarts illustrent la "
        "diversite de sentiments apportee par le dataset combine."
    )

with tab_models:
    st.subheader("Comparaison des modeles entraines")
    metrics = load_model_metrics()

    comparison = pd.DataFrame(
        {
            name: {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
            for name, m in metrics.items()
        }
    ).T
    st.dataframe(comparison.style.format("{:.3f}"))
    st.bar_chart(comparison)
    best_model = comparison["macro_f1"].idxmax()
    st.caption(
        "L'accuracy seule serait trompeuse ici (le dataset est desequilibre) -- le "
        "macro F1 (moyenne par classe, sans favoriser la classe majoritaire) est la "
        f"metrique de reference. '{best_model}' obtient le meilleur macro F1."
    )

    st.warning(
        "Le sentiment 'neutre' reste le plus difficile a apprendre : il est "
        "naturellement sous-represente dans les avis reels (peu de clients "
        "laissent un avis explicitement neutre). Voir les matrices de "
        "confusion ci-dessous et `notebooks/03_modeling.ipynb`."
    )

    st.write("**Matrices de confusion (jeu de test)**")
    items = list(metrics.items())
    cols_per_row = 3
    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (name, m) in zip(cols, row_items):
            with col:
                st.write(name)
                labels = m.get("labels", ["negative", "neutral", "positive"])
                matrix = pd.DataFrame(
                    m["confusion_matrix"], index=labels, columns=labels
                )
                st.dataframe(matrix)
    st.caption(
        "Comment lire une matrice de confusion : chaque ligne est le sentiment reel, "
        "chaque colonne le sentiment predit. Les valeurs sur la diagonale sont les "
        "bonnes predictions ; tout ce qui est hors diagonale est une erreur. La classe "
        "'neutral' est rarement predite correctement car elle est sous-representee "
        "dans les donnees."
    )
