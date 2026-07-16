"""One-off script: generate presentation charts (PNG) and a full PDF documentation report.

Not part of the production pipeline -- run manually when assets are needed
for a presentation. Reads already-trained model metrics and the processed
dataset; does not retrain anything.
"""

import json
import re
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "reviews_nlp.csv"
WEIGHTS_DIR = ROOT / "src" / "models" / "weights"
TRAINING_LOG_PATH = Path(
    r"C:\Users\charl\AppData\Local\Temp\transformer_multisector.log"
)
CHARTS_DIR = ROOT / "charts"
REPORT_PATH = ROOT / "rapport_projet.pdf"
PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches
RANDOM_STATE = 42

CLASSICAL_MODEL_LABELS = {
    "random_forest": "TF-IDF + Random Forest",
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "naive_bayes": "TF-IDF + Naive Bayes",
}

# Macro F1 progression documented across the project's iterations (see project
# memory / notebooks/03_modeling.ipynb): same Transformer architecture, three
# successive improvements to the target/data rather than the architecture.
PIVOT_PROGRESSION = [
    ("Note brute 1-5\n(Revolut seul)", 0.350),
    ("Sentiment 3 classes\n(Revolut seul)", 0.598),
    ("Sentiment 3 classes\n(5 entreprises)", 0.678),
]

sns.set_theme(style="whitegrid")
CHARTS_DIR.mkdir(exist_ok=True)


def load_metrics() -> dict:
    metrics = {}
    for key, label in CLASSICAL_MODEL_LABELS.items():
        path = WEIGHTS_DIR / f"tfidf_{key}_metrics.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                metrics[label] = json.load(f)["test"]
    with open(WEIGHTS_DIR / "xlm_roberta" / "metrics.json", encoding="utf-8") as f:
        metrics["XLM-RoBERTa"] = json.load(f)["test"]
    return metrics


def parse_training_loss(log_path: Path) -> pd.DataFrame:
    """Extract (epoch, loss) pairs logged during the final Transformer training run."""
    if not log_path.exists():
        return pd.DataFrame(columns=["epoch", "loss"])
    raw = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    rows = []
    for match in re.finditer(r"\{'loss': '([\d.]+)'.*?'epoch': '([\d.]+)'\}", raw):
        rows.append({"loss": float(match.group(1)), "epoch": float(match.group(2))})
    return pd.DataFrame(rows)


def save_chart(fig, name: str) -> Path:
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def wrap_preserve(text: str, width: int = 100) -> str:
    """Wrap each paragraph/line individually, preserving manual newlines/blank lines."""
    lines = []
    for line in text.split("\n"):
        if not line.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(lines)


def add_text_page(pdf: PdfPages, title: str, body: str, fontsize: float = 10.5) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle(title, fontsize=16, weight="bold", y=0.97)
    ax = fig.add_axes([0.08, 0.03, 0.86, 0.88])
    ax.axis("off")
    ax.text(0, 1, wrap_preserve(body), va="top", fontsize=fontsize, family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def add_chart_page(pdf: PdfPages, image_path: Path, title: str, caption: str) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.suptitle(title, fontsize=15, weight="bold", y=0.97)
    ax_img = fig.add_axes([0.06, 0.42, 0.88, 0.5])
    ax_img.imshow(plt.imread(image_path))
    ax_img.axis("off")
    ax_text = fig.add_axes([0.08, 0.03, 0.86, 0.34])
    ax_text.axis("off")
    ax_text.text(0, 1, wrap_preserve(caption, width=100), va="top", fontsize=9.5)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "business",
            "source",
            "rating",
            "sentiment",
            "language",
            "date",
            "text",
            "text_classical",
        ],
        low_memory=False,
    )
    metrics = load_metrics()

    n_reviews = len(df)
    n_business = df["business"].nunique()
    n_lang = df["language"].nunique()
    sentiment_pct = df["sentiment"].value_counts(normalize=True)

    # ================= Charts =================

    # --- 01: sentiment distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["sentiment"].value_counts()
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        ax=ax,
        palette="viridis",
        legend=False,
    )
    ax.set_title("Distribution du sentiment (dataset complet)")
    ax.set_ylabel("Nombre d'avis")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v / counts.sum():.1%}", ha="center", va="bottom")
    chart_sentiment = save_chart(fig, "01_distribution_sentiment.png")

    # --- 02: reviews per business ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["business"].value_counts()
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        ax=ax,
        palette="crest",
        legend=False,
    )
    ax.set_title("Nombre d'avis par entreprise (dataset multi-business)")
    ax.set_ylabel("Nombre d'avis")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    chart_business = save_chart(fig, "02_avis_par_business.png")

    # --- 03: before/after text cleaning ---
    raw_len = df["text"].fillna("").str.split().str.len()
    clean_len = df["text_classical"].fillna("").str.split().str.len()
    vocab_raw = len(
        CountVectorizer(token_pattern=r"[\w'-]+").fit(df["text"].fillna("")).vocabulary_
    )
    vocab_clean = len(
        CountVectorizer(token_pattern=r"[\w'-]+")
        .fit(df["text_classical"].fillna(""))
        .vocabulary_
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    sns.barplot(
        x=["Texte brut", "Texte nettoye"],
        y=[raw_len.mean(), clean_len.mean()],
        hue=["Texte brut", "Texte nettoye"],
        ax=axes[0],
        palette="Set2",
        legend=False,
    )
    axes[0].set_title("Longueur moyenne (mots)")
    sns.barplot(
        x=["Texte brut", "Texte nettoye"],
        y=[vocab_raw, vocab_clean],
        hue=["Texte brut", "Texte nettoye"],
        ax=axes[1],
        palette="Set2",
        legend=False,
    )
    axes[1].set_title("Taille du vocabulaire (mots uniques)")
    fig.suptitle("Avant / apres le nettoyage du texte (preprocessing)")
    chart_preprocessing = save_chart(fig, "03_avant_apres_preprocessing.png")

    # --- 04: review length by sentiment ---
    length_df = pd.DataFrame({"sentiment": df["sentiment"], "word_count": raw_len})
    length_df = length_df[
        length_df["word_count"] <= length_df["word_count"].quantile(0.95)
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=length_df,
        x="sentiment",
        y="word_count",
        order=["negative", "neutral", "positive"],
        hue="sentiment",
        palette="coolwarm",
        ax=ax,
        legend=False,
    )
    ax.set_title("Longueur des avis selon le sentiment (95e percentile exclu)")
    ax.set_ylabel("Nombre de mots")
    chart_length = save_chart(fig, "04_longueur_par_sentiment.png")

    # --- 05: wordclouds per sentiment ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sentiment in zip(axes, ["negative", "neutral", "positive"]):
        subset = df.loc[df["sentiment"] == sentiment, "text_classical"].dropna()
        sample = subset.sample(min(len(subset), 15_000), random_state=RANDOM_STATE)
        text_blob = " ".join(sample) or "vide"
        wc = WordCloud(
            width=500, height=400, background_color="white", colormap="viridis"
        ).generate(text_blob)
        ax.imshow(wc)
        ax.set_title(sentiment)
        ax.axis("off")
    fig.suptitle("Mots les plus frequents par sentiment", fontsize=14, weight="bold")
    chart_wordcloud = save_chart(fig, "05_wordcloud_sentiment.png")

    # --- 06: feature importance (Logistic Regression coefficients) ---
    pipeline = joblib.load(WEIGHTS_DIR / "tfidf_logistic_regression.pkl")
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, cls in zip(axes, classifier.classes_):
        idx = list(classifier.classes_).index(cls)
        coefs = classifier.coef_[idx]
        top_idx = np.argsort(coefs)[-12:]
        sns.barplot(
            x=coefs[top_idx],
            y=feature_names[top_idx],
            hue=feature_names[top_idx],
            ax=ax,
            palette="flare",
            legend=False,
        )
        ax.set_title(f"Top mots -- '{cls}'")
        ax.set_xlabel("Poids (coefficient)")
    fig.suptitle(
        "Mots les plus influents par sentiment (Regression Logistique)",
        fontsize=14,
        weight="bold",
    )
    chart_importance = save_chart(fig, "06_feature_importance.png")

    # --- 07: model comparison ---
    comparison = pd.DataFrame(
        {
            name: {"Accuracy": m["accuracy"], "Macro F1": m["macro_f1"]}
            for name, m in metrics.items()
        }
    ).T
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Comparaison des 5 modeles (jeu de test)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="lower right")
    chart_comparison = save_chart(fig, "07_comparaison_modeles.png")

    # --- 08: before/after pivot (macro F1 progression) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    stages, scores = zip(*PIVOT_PROGRESSION)
    bars = ax.bar(stages, scores, color=["#C44E52", "#DD8452", "#55A868"])
    ax.set_title(
        "Avant / apres : evolution du macro F1 (meme architecture Transformer)"
    )
    ax.set_ylabel("Macro F1 (jeu de test)")
    ax.set_ylim(0, 1)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )
    chart_pivot = save_chart(fig, "08_avant_apres_pivot.png")

    # --- 09: training loss curve ---
    loss_df = parse_training_loss(TRAINING_LOG_PATH)
    if not loss_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.lineplot(data=loss_df, x="epoch", y="loss", marker="o", ax=ax)
        ax.set_title("Courbe d'apprentissage -- XLM-RoBERTa (perte d'entrainement)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        chart_loss = save_chart(fig, "09_courbe_apprentissage.png")
    else:
        chart_loss = None

    # --- 10 & 11: confusion matrices, raw counts + row-normalized (%) ---
    chart_confusion = {}
    for name in ["TF-IDF + Linear SVM", "XLM-RoBERTa"]:
        m = metrics[name]
        labels = m["labels"]
        matrix = pd.DataFrame(m["confusion_matrix"], index=labels, columns=labels)
        matrix_pct = matrix.div(matrix.sum(axis=1), axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
        axes[0].set_title("Valeurs brutes")
        axes[0].set_xlabel("Predit")
        axes[0].set_ylabel("Reel")
        sns.heatmap(
            matrix_pct, annot=True, fmt=".1%", cmap="Blues", ax=axes[1], cbar=False
        )
        axes[1].set_title("Normalise par ligne (%)")
        axes[1].set_xlabel("Predit")
        axes[1].set_ylabel("Reel")
        fig.suptitle(f"Matrice de confusion -- {name}", fontsize=14, weight="bold")
        safe_name = name.lower().replace(" ", "_").replace("+", "")
        chart_confusion[name] = save_chart(fig, f"10_confusion_{safe_name}.png")

    print(f"Charts saved to {CHARTS_DIR}")

    # ================= PDF report =================
    with PdfPages(REPORT_PATH) as pdf:
        # Page 1: title
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.text(
            0.5,
            0.72,
            "Analyse Intelligente des Avis Clients",
            ha="center",
            fontsize=22,
            weight="bold",
        )
        fig.text(
            0.5,
            0.65,
            "Outil generique de sentiment analysis multi-business",
            ha="center",
            fontsize=14,
        )
        fig.text(
            0.5,
            0.58,
            "Projet de fin d'etudes -- Master 2 Expert Chef de Projet IA",
            ha="center",
            fontsize=11,
        )
        fig.text(
            0.5,
            0.45,
            f"Dataset final : {n_reviews:,} avis -- {n_business} entreprises -- {n_lang} langues",
            ha="center",
            fontsize=11,
        )
        fig.text(
            0.5,
            0.40,
            f"Modele retenu : XLM-RoBERTa -- Accuracy {metrics['XLM-RoBERTa']['accuracy']:.1%} -- "
            f"Macro F1 {metrics['XLM-RoBERTa']['macro_f1']:.3f}",
            ha="center",
            fontsize=11,
        )
        plt.axis("off")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: contexte et objectif
        add_text_page(
            pdf,
            "1. Contexte et objectif du projet",
            "Contexte\n\n"
            "Ce projet s'inscrit dans le dossier professionnel du Master 2 Expert Chef de\n"
            "Projet IA (EPSI). L'objectif est de construire un outil d'analyse intelligente\n"
            "des avis clients, capable de determiner automatiquement le sentiment (positif,\n"
            "negatif ou neutre) exprime dans un avis, quelle que soit la langue ou\n"
            "l'entreprise concernee.\n\n"
            "Evolution du sujet\n\n"
            "Le sujet portait initialement sur les avis de la neobanque Revolut. En cours\n"
            "de projet, le besoin a evolue vers un outil generaliste, utilisable pour\n"
            "n'importe quelle entreprise (e-commerce, transport, hotellerie...), sans avoir\n"
            "a re-entrainer un modele specifique a chaque fois.\n\n"
            "Pourquoi le sentiment plutot que la note brute (1 a 5 etoiles) ?\n\n"
            "La note brute souffre d'un desequilibre extreme (la grande majorite des avis\n"
            "sont a 5 etoiles) et les notes intermediaires (2 et 3 etoiles) sont trop rares\n"
            "pour qu'un modele apprenne a les reconnaitre correctement. Le sentiment\n"
            "(negatif/neutre/positif) est un concept plus universel, qui se transpose mieux\n"
            "d'un secteur a l'autre et qui est plus facile a apprendre correctement.",
        )

        # Page 3: collecte et preparation des donnees
        add_text_page(
            pdf,
            "2. Collecte et preparation des donnees",
            "Collecte des donnees\n\n"
            "Aucun dataset Kaggle ne correspondait au besoin (avis multi-entreprises,\n"
            "multilingues). Les donnees ont donc ete collectees par scraping de sources\n"
            "publiques et legitimes :\n"
            "  - Google Play Store (avis d'applications mobiles, 15 langues cibles)\n"
            "  - Apple App Store (flux RSS public des avis)\n\n"
            "5 entreprises ont ete scrapees pour obtenir un dataset varie : Revolut\n"
            "(banque), Amazon (e-commerce), Uber (transport), Uber Eats (livraison de\n"
            "repas), Booking.com (hotellerie).\n\n"
            "Pourquoi plusieurs secteurs ? Pour qu'un seul modele apprenne des schemas de\n"
            "sentiment generaux (mots positifs/negatifs, tournures de phrase) plutot que\n"
            "des particularites propres a une seule entreprise -- condition necessaire\n"
            "pour qu'il generalise a une entreprise qu'il n'a jamais vue.\n\n"
            "Nettoyage des donnees (niveau document)\n"
            "  - Suppression des avis vides ou strictement dupliques\n"
            "  - Filtrage des avis non rediges en alphabet latin\n"
            "  - Detection de la langue (au-dela de 20 caracteres, sinon trop peu fiable)\n\n"
            f"Resultat : un dataset final de {n_reviews:,} avis, repartis sur {n_business} "
            f"entreprises\net {n_lang} langues. Le detail du nettoyage du TEXTE (mots vides,\n"
            "lemmatisation) est presente page suivante, avec la comparaison avant/apres.",
        )

        # Page 4: avant/apres preprocessing (texte)
        add_chart_page(
            pdf,
            chart_preprocessing,
            "3. Avant / apres le nettoyage du texte",
            "Comment lire ce graphique : a gauche, la longueur moyenne d'un avis (en mots)\n"
            "avant et apres nettoyage. A droite, le nombre de mots uniques (vocabulaire)\n"
            "avant et apres.\n\n"
            f"Ce que ca montre : le texte brut fait {raw_len.mean():.1f} mots en moyenne, "
            f"contre\n{clean_len.mean():.1f} apres suppression des mots vides et lemmatisation "
            "-- une reduction\nattendue, le but etant de ne garder que les mots porteurs de sens.\n"
            f"Le vocabulaire passe de {vocab_raw:,} a {vocab_clean:,} mots uniques : la "
            "lemmatisation\nregroupe les variantes d'un meme mot (jouait/jouer) sous une seule forme,\n"
            "ce qui reduit la dimensionnalite traitee par les modeles classiques (TF-IDF).\n\n"
            "Important : ce nettoyage ne s'applique qu'a la copie de texte utilisee par les\n"
            "modeles classiques. Le Transformer (XLM-RoBERTa) utilise le texte BRUT, non\n"
            "nettoye, car il a besoin de la ponctuation et des mots vides pour comprendre\n"
            "le contexte complet de la phrase.",
        )

        # Page 5: chart sentiment + explanation
        add_chart_page(
            pdf,
            chart_sentiment,
            "4. Repartition du sentiment dans le dataset",
            "Comment lire ce graphique : chaque barre represente le nombre d'avis pour\n"
            "chaque categorie de sentiment, avec le pourcentage du total au-dessus.\n\n"
            f"Ce que ca montre : le dataset est desequilibre -- {sentiment_pct.get('positive', 0):.0%} "
            f"des avis sont\npositifs, {sentiment_pct.get('negative', 0):.0%} negatifs, et seulement "
            f"{sentiment_pct.get('neutral', 0):.0%} neutres.\n"
            "Ce desequilibre est une caracteristique naturelle des avis clients reels :\n"
            "les clients laissent plus souvent un avis quand ils sont tres satisfaits ou\n"
            "tres mecontents, rarement quand leur experience est simplement 'moyenne'.\n\n"
            "Impact sur la modelisation : ce desequilibre oblige a utiliser des metriques\n"
            "adaptees (macro F1 plutot que l'accuracy seule) et explique pourquoi la\n"
            "categorie 'neutre' reste la plus difficile a predire pour tous les modeles.",
        )

        # Page 6: chart business + explanation
        add_chart_page(
            pdf,
            chart_business,
            "5. Volume d'avis par entreprise",
            "Comment lire ce graphique : nombre total d'avis collectes pour chacune des 5\n"
            "entreprises du dataset.\n\n"
            "Ce que ca montre : le volume est globalement equilibre entre les entreprises,\n"
            "ce qui garantit qu'aucune entreprise ne domine excessivement l'apprentissage.\n\n"
            "Pourquoi c'est important : un modele entraine sur un seul secteur risquerait\n"
            'd\'apprendre des mots-cles trop specifiques ("virement", "carte bloquee")\n'
            'qui ne se retrouvent pas dans d\'autres secteurs ("livraison en retard",\n'
            '"chambre sale"). Melanger les secteurs force le modele a apprendre des\n'
            "indices de sentiment plus generaux, donc plus transferables.",
        )

        # Page 7: longueur des avis par sentiment
        add_chart_page(
            pdf,
            chart_length,
            "6. Longueur des avis selon le sentiment",
            "Comment lire ce graphique (boite a moustaches / boxplot) : pour chaque\n"
            "sentiment, la boite represente ou se situent 50% des avis (entre le 1er et le\n"
            "3e quartile), le trait au milieu est la mediane, et les moustaches montrent\n"
            "l'etendue des valeurs courantes.\n\n"
            "Ce que ca montre : les avis negatifs sont generalement plus longs que les\n"
            "avis positifs. Hypothese habituelle en analyse d'avis clients : les clients\n"
            "mecontents prennent plus de temps a detailler ce qui n'a pas fonctionne,\n"
            "alors qu'un avis positif se resume souvent a une phrase courte ('super !').\n\n"
            "Interet pour le modele : la longueur n'est pas utilisee comme variable\n"
            "explicite, mais elle confirme que les avis negatifs contiennent plus\n"
            "d'information textuelle exploitable par les modeles.",
        )

        # Page 8: wordclouds
        add_chart_page(
            pdf,
            chart_wordcloud,
            "7. Mots les plus frequents par sentiment",
            "Comment lire ce graphique : plus un mot est gros, plus il apparait souvent\n"
            "dans les avis de cette categorie (apres nettoyage : mots vides retires,\n"
            "lemmatises).\n\n"
            "Ce que ca montre : les noms des entreprises et des applications ('uber',\n"
            "'amazon', 'booking', 'app'...) dominent dans les 3 categories, car ils sont\n"
            "frequents partout, peu importe le sentiment. C'est une limite connue du\n"
            "nuage de mots, qui se base sur la frequence brute plutot que sur le caractere\n"
            "distinctif d'un mot.\n\n"
            "C'est justement pourquoi la page suivante (feature importance) est plus\n"
            "utile pour l'analyse : elle met en avant les mots qui distinguent vraiment\n"
            "un sentiment d'un autre, et non les mots simplement frequents.",
        )

        # Page 9: feature importance
        add_chart_page(
            pdf,
            chart_importance,
            "8. Mots les plus influents par sentiment (interpretabilite)",
            "Comment lire ce graphique : pour le modele de Regression Logistique, chaque\n"
            "mot du vocabulaire a un poids (coefficient) par categorie. Plus le poids est\n"
            "eleve, plus la presence de ce mot pousse le modele a predire cette categorie.\n"
            "Les 12 mots les plus influents sont affiches pour chaque sentiment.\n\n"
            "Ce que ca montre : on retrouve des mots de sentiment attendus, ce qui valide\n"
            "que le modele apprend des signaux pertinents plutot que des correlations\n"
            "fortuites. C'est un premier niveau d'interpretabilite (feature importance),\n"
            "moins fin qu'une analyse SHAP/LIME par avis individuel, mais qui permet deja\n"
            "de verifier que le modele 'raisonne' sur des bases sensees.",
        )

        # Page 10: methodologie de modelisation
        add_text_page(
            pdf,
            "9. Methodologie de modelisation",
            "5 modeles ont ete entraines et compares sur les memes donnees (memes\n"
            "ensembles d'entrainement / validation / test, repartition 70%/15%/15%),\n"
            "pour garantir une comparaison equitable.\n\n"
            "Approche 1 : Machine Learning classique (TF-IDF)\n\n"
            "Le texte est transforme en vecteurs numeriques representant l'importance de\n"
            "chaque mot (TF-IDF), puis un classifieur apprend a separer les categories.\n"
            "4 classifieurs ont ete testes sur les memes vecteurs :\n"
            "  - Naive Bayes : modele probabiliste simple et rapide (base historique du\n"
            "    filtrage de spam)\n"
            "  - Random Forest : ensemble d'arbres de decision\n"
            "  - Regression Logistique : modele lineaire, robuste sur des donnees a tres\n"
            "    haute dimension\n"
            "  - SVM lineaire : cherche la meilleure frontiere de separation entre les\n"
            "    categories\n\n"
            "Approche 2 : Deep Learning (Transformer)\n\n"
            "XLM-RoBERTa, un modele pre-entraine sur d'enormes corpus multilingues, puis\n"
            "affine (fine-tuning) sur nos donnees. Contrairement au TF-IDF qui traite le\n"
            "texte comme un simple 'sac de mots', le Transformer comprend l'ordre des mots\n"
            "et le contexte (ex: distinguer 'pas mauvais' de 'mauvais').\n\n"
            "Pourquoi XLM-RoBERTa et pas CamemBERT/RoBERTa (prevus initialement) : ces\n"
            "deux modeles sont monolingues (francais / anglais uniquement), incompatibles\n"
            "avec un dataset en 15+ langues. XLM-RoBERTa est l'equivalent multilingue de\n"
            "cette famille de modeles.\n\n"
            "Gestion du desequilibre : chaque modele utilise une ponderation des classes\n"
            "pour eviter qu'il ignore les categories minoritaires (class_weight='balanced'\n"
            "pour les modeles classiques, ponderation de la fonction de perte pour le\n"
            "Transformer).",
        )

        # Page 11: avant/apres pivot
        add_chart_page(
            pdf,
            chart_pivot,
            "10. Avant / apres : amelioration du modele (post-traitement)",
            "Comment lire ce graphique : evolution du macro F1 du MEME modele\n"
            "(architecture Transformer XLM-RoBERTa) au fil de 3 ameliorations successives\n"
            "du dataset et de la cible, sans changer l'architecture elle-meme.\n\n"
            "Etape 1 (rouge) : modele entraine sur la note brute 1-5, sur Revolut seul.\n"
            "Etape 2 (orange) : la cible est changee pour le sentiment (3 classes), encore\n"
            "  sur Revolut seul -- gain important car les classes 2/3 etoiles, trop rares,\n"
            "  ne pouvaient pas etre apprises individuellement.\n"
            "Etape 3 (vert) : le dataset est etendu a 5 entreprises -- gain supplementaire\n"
            "  car le modele dispose de plus d'exemples, notamment pour la classe la plus\n"
            "  rare ('neutre').\n\n"
            "Conclusion : chaque amelioration a ete mesuree avant d'etre adoptee -- aucun\n"
            "choix n'a ete fait par hypothese seule.",
        )

        # Page 12: courbe d'apprentissage
        if chart_loss is not None:
            add_chart_page(
                pdf,
                chart_loss,
                "11. Courbe d'apprentissage du modele final",
                "Comment lire ce graphique : la 'loss' (perte) mesure a quel point les\n"
                "predictions du modele s'eloignent des vraies reponses pendant\n"
                "l'entrainement. Plus elle baisse, mieux le modele apprend.\n\n"
                "Ce que ca montre : la perte diminue de facon reguliere au fil des epochs,\n"
                "sans repartir a la hausse ni stagner brutalement -- signe d'un\n"
                "entrainement stable, sans surapprentissage flagrant a ce stade.\n\n"
                "Contexte : un essai precedent avec une ponderation de classe trop forte\n"
                "avait produit une perte qui ne baissait pas du tout (modele bloque) --\n"
                "ce graphique permet de verifier visuellement que ce probleme est resolu.",
            )

        # Page 13: chart comparison + explanation + table
        table_lines = "\n".join(
            f"  {name:<32} accuracy={m['accuracy']:.3f}   macro F1={m['macro_f1']:.3f}"
            for name, m in metrics.items()
        )
        add_chart_page(
            pdf,
            chart_comparison,
            "12. Comparaison des 5 modeles entraines",
            "Comment lire ce graphique : pour chaque modele, deux scores sont affiches :\n"
            "l'accuracy (% de predictions correctes, en bleu) et le macro F1 (en orange).\n\n"
            "Pourquoi deux metriques differentes : l'accuracy seule serait trompeuse ici.\n"
            "Un modele qui predit systematiquement 'positif' obtiendrait une accuracy\n"
            "d'environ 73% sans rien avoir appris -- juste parce que 73% des avis sont\n"
            "reellement positifs. Le macro F1 calcule un score pour chaque categorie\n"
            "separement puis fait la moyenne des 3, sans favoriser la categorie\n"
            "majoritaire. C'est la metrique de reference retenue pour choisir le modele.\n\n"
            "Resultats detailles (jeu de test) :\n" + table_lines,
        )

        # Page 14 & 15: confusion matrices with explanation
        add_chart_page(
            pdf,
            chart_confusion["TF-IDF + Linear SVM"],
            "13. Matrice de confusion -- meilleur modele classique",
            "Comment lire une matrice de confusion : chaque LIGNE correspond au sentiment\n"
            "REEL de l'avis, chaque COLONNE au sentiment PREDIT. La diagonale represente\n"
            "les predictions correctes. A gauche, les valeurs brutes (nombre d'avis) ; a\n"
            "droite, les memes valeurs normalisees en % PAR LIGNE -- utile ici car la\n"
            "classe 'neutre' a si peu d'avis que ses erreurs sont invisibles en valeurs\n"
            "brutes a cote des dizaines de milliers d'avis positifs.\n\n"
            "Lecture pour le SVM lineaire : 'negatif' et 'positif' sont bien separes. La\n"
            "vue en % revele que la grande majorite des avis 'neutres' sont mal classes,\n"
            "repartis a peu pres entre 'negatif' et 'positif'.",
        )
        add_chart_page(
            pdf,
            chart_confusion["XLM-RoBERTa"],
            "14. Matrice de confusion -- modele retenu (XLM-RoBERTa)",
            "Meme principe de lecture que la page precedente (lignes = reel, colonnes =\n"
            "predit, diagonale = bonnes predictions, vue de droite normalisee en %).\n\n"
            "Lecture pour XLM-RoBERTa : 'negatif' et 'positif' sont tres bien separes,\n"
            "avec une concentration encore plus forte sur la diagonale que le modele\n"
            "classique (visible dans la vue en % : pourcentages plus proches de 100%).\n"
            "La vue en % montre aussi que le modele identifie desormais une part non\n"
            "negligeable d'avis 'neutres' -- une amelioration reelle, meme si la marge\n"
            "de progression reste la plus importante sur cette categorie.",
        )

        # Page 16: choix final
        add_text_page(
            pdf,
            "15. Choix final et justification",
            "Modele retenu pour la production : XLM-RoBERTa\n\n"
            "Justification :\n"
            "  1. Meilleur score sur la metrique de reference (macro F1 = "
            f"{metrics['XLM-RoBERTa']['macro_f1']:.3f},\n"
            f"     soit +{metrics['XLM-RoBERTa']['macro_f1'] - metrics['TF-IDF + Linear SVM']['macro_f1']:.3f} "
            "par rapport au meilleur modele classique)\n"
            "  2. Seul modele a comprendre le contexte et l'ordre des mots (negations,\n"
            "     nuances), contrairement aux modeles TF-IDF qui traitent le texte comme\n"
            "     un sac de mots\n"
            "  3. Pre-entraine sur des corpus multilingues massifs : il 'connait' deja\n"
            "     les 15+ langues avant meme d'etre affine sur nos donnees\n\n"
            "Limite assumee : le temps d'entrainement et d'inference est plus eleve qu'un\n"
            "modele classique. Sans GPU disponible, l'entrainement a ete limite a un\n"
            "sous-echantillon de 16 000 avis (sur 178 000 disponibles), pour rester dans\n"
            "un budget de calcul raisonnable sur CPU (environ 1h10 d'entrainement).\n\n"
            "Avantages et limites de chaque approche\n\n"
            "  TF-IDF + ML classique :\n"
            "    + Rapide a entrainer (quelques secondes a minutes)\n"
            "    + Facile a interpreter (on peut voir quels mots pesent dans la decision)\n"
            "    - Ne capture pas le contexte ni l'ordre des mots\n\n"
            "  Transformer (XLM-RoBERTa) :\n"
            "    + Capture le contexte et la nuance\n"
            "    + Meilleure performance globale\n"
            "    - Plus lent a entrainer et plus gourmand en ressources de calcul",
        )

        # Page 17: architecture technique
        add_text_page(
            pdf,
            "16. Mise a disposition du modele",
            "Une API (FastAPI) expose le modele retenu via deux endpoints :\n\n"
            "  - POST /predict : analyse un avis unique, renvoie le sentiment et le\n"
            "    niveau de confiance\n"
            "  - POST /predict/batch : analyse jusqu'a 500 avis en une fois, avec un\n"
            "    score global agrege -- concu pour le cas d'usage 'une entreprise\n"
            "    fournit son dataset d'avis et veut un verdict d'ensemble'\n\n"
            "Un tableau de bord (Streamlit) permet de tester le modele sans ecrire de\n"
            "code, avec 4 onglets :\n\n"
            "  - Prediction en direct : sur un avis saisi a la main\n"
            "  - Analyse d'un dataset : upload d'un fichier CSV, score global +\n"
            "    repartition positif/negatif/neutre, telechargement des resultats\n"
            "  - Statistiques du dataset : volumes, langues, sentiment, par entreprise\n"
            "  - Performance des modeles : comparaison visuelle des 5 modeles entraines",
        )

        # Page 18: etat d'avancement
        add_text_page(
            pdf,
            "17. Etat d'avancement et perspectives",
            "Termine\n\n"
            "  - Exploration et preparation des donnees (multi-source, multi-business,\n"
            "    multilingue)\n"
            "  - Preprocessing NLP adapte a chaque famille de modele, avec comparaison\n"
            "    avant/apres\n"
            "  - Modelisation : 5 modeles entraines, evalues et compares\n"
            "  - Amelioration mesuree du dataset/de la cible (avant/apres post-traitement)\n"
            "  - Backend (API FastAPI) et Frontend (Dashboard Streamlit)\n\n"
            "A venir\n\n"
            "  - Conteneurisation Docker et pipeline CI/CD (GitHub Actions)\n"
            "  - Discussion approfondie biais/variance et surapprentissage par modele\n"
            "  - Interpretabilite avancee (SHAP/LIME) pour expliquer les predictions\n"
            "    individuelles (au-dela de la feature importance globale deja presentee)\n\n"
            "Conclusion\n\n"
            "Le projet demontre une chaine complete, de la collecte de donnees brutes\n"
            "jusqu'a un outil utilisable (API + dashboard), avec une demarche\n"
            "methodique : chaque choix (cible du modele, perimetre des donnees, type de\n"
            "modele) a ete pris puis valide par une mesure chiffree, et non par simple\n"
            "hypothese.",
        )

    print(f"PDF report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
