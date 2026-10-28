from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_regression_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"
DATASET_ID = "hasanyiitakbulut/ai-and-human-text-dataset"
MAX_DATASET_ROWS = 100_000

st.set_page_config(
    page_title="TextLens | AI & Human Text",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #f4f5ff;
        --muted: #9ca3bd;
        --panel: #14172a;
        --line: rgba(184, 190, 225, .12);
        --violet: #9b8cff;
        --cyan: #72e1db;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% -8%, rgba(102, 82, 201, .20), transparent 34%),
            radial-gradient(ellipse at 0% 38%, rgba(33, 130, 151, .09), transparent 30%),
            #0c0e19;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(12, 14, 25, .75); }
    [data-testid="stSidebar"] {
        background: rgba(15, 17, 31, .96);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.7rem; }
    h1, h2, h3, h4 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }
    h1 { font-size: clamp(2.1rem, 4vw, 3.25rem) !important; line-height: 1.1 !important; }
    p, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
    .brand {
        color: #f4f5ff; font-family: 'Manrope', sans-serif; font-size: 1.3rem;
        font-weight: 800; letter-spacing: -.05em; padding: .3rem 0 1.8rem;
    }
    .brand-mark {
        display: inline-grid; place-items: center; width: 31px; height: 31px;
        margin-right: 9px; border-radius: 10px; color: #111326;
        background: linear-gradient(135deg, #b6aaff, #75e1d6); font-size: 1rem;
    }
    .eyebrow {
        color: var(--cyan); font-size: .72rem; font-weight: 700;
        letter-spacing: .16em; text-transform: uppercase; margin-bottom: .65rem;
    }
    .hero {
        position: relative; overflow: hidden; padding: 2.1rem 2.25rem;
        margin: .5rem 0 1.5rem; border-radius: 24px;
        border: 1px solid rgba(183, 171, 255, .22);
        background: linear-gradient(115deg, rgba(48, 43, 88, .9), rgba(22, 31, 50, .92) 70%, rgba(23, 49, 59, .85));
        box-shadow: 0 24px 80px rgba(0, 0, 0, .20);
    }
    .hero:after {
        content: ""; position: absolute; width: 260px; height: 260px; right: 4%;
        top: -145px; border-radius: 50%; border: 1px solid rgba(192, 183, 255, .22);
        box-shadow: 0 0 0 28px rgba(192,183,255,.035), 0 0 0 58px rgba(192,183,255,.025);
    }
    .hero p { max-width: 670px; color: #c3c6dc; line-height: 1.7; margin: .7rem 0 0; }
    .panel {
        background: linear-gradient(145deg, rgba(23, 26, 45, .96), rgba(18, 21, 36, .96));
        border: 1px solid var(--line); border-radius: 19px; padding: 1.25rem 1.35rem;
    }
    .metric-label { color: var(--muted); font-size: .8rem; margin-bottom: .5rem; }
    .metric-value { font: 700 1.55rem 'Manrope', sans-serif; color: #f5f5ff; }
    .metric-foot { color: #8189a6; font-size: .74rem; margin-top: .35rem; }
    .section-note { color: var(--muted); margin-top: -.5rem; margin-bottom: 1rem; }
    .soft-divider { border: 0; border-top: 1px solid var(--line); margin: 1.3rem 0; }
    .result-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 11px; border: 1px solid rgba(178, 167, 255, .42);
        background: linear-gradient(110deg, #9b8cff, #75dcd6); color: #111326;
        font-weight: 700; transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); filter: brightness(1.08); color: #111326; }
    div[data-baseweb="textarea"] > div, div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #111425; border-color: var(--line); border-radius: 10px;
    }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(23, 26, 45, .96), rgba(18, 21, 36, .96));
        border: 1px solid var(--line); padding: 1rem 1.15rem; border-radius: 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Manrope', sans-serif; }
    .stRadio [role="radiogroup"] { gap: .25rem; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> tuple[Any, Any, Any]:
    missing = [
        path.name for path in (MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required model file(s) are missing: "
            + ", ".join(missing)
            + ". Place all three .joblib files beside app.py."
        )
    return (
        joblib.load(MODEL_PATH),
        joblib.load(VECTORIZER_PATH),
        joblib.load(ENCODER_PATH),
    )


@st.cache_data(show_spinner="Fetching the latest Kaggle dataset…")
def load_dataset() -> tuple[pd.DataFrame, str]:
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    supported = {".csv", ".parquet", ".json", ".jsonl"}
    files = sorted(
        file for file in dataset_path.rglob("*") if file.is_file() and file.suffix.lower() in supported
    )
    if not files:
        raise FileNotFoundError(f"No CSV, Parquet, or JSON data files found in {dataset_path}.")

    frames: list[pd.DataFrame] = []
    for file in files:
        if file.suffix.lower() == ".csv":
            frame = pd.read_csv(file, nrows=MAX_DATASET_ROWS, low_memory=False)
        elif file.suffix.lower() == ".parquet":
            frame = pd.read_parquet(file)
        else:
            frame = pd.read_json(file, lines=file.suffix.lower() == ".jsonl")
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(f"The dataset files in {dataset_path} contain no rows.")

    return pd.concat(frames, ignore_index=True, sort=False), ", ".join(
        file.name for file in files
    )


def find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    by_lowercase = {str(column).strip().lower(): str(column) for column in frame.columns}
    return next((by_lowercase[name.lower()] for name in names if name.lower() in by_lowercase), None)


def get_display_label(value: Any, encoder: Any) -> str:
    try:
        return str(encoder.inverse_transform([int(value)])[0])
    except (TypeError, ValueError, IndexError):
        return str(value)


def render_header(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f"""
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p class="section-note">{subtitle}</p>
        """,
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Language intelligence studio</div>
          <h1>See the signal<br>behind the sentence.</h1>
          <p>Explore human and AI-authored writing, inspect the dataset, and run your text through the trained classifier — all in one workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    present = all(path.is_file() for path in (MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="panel"><div class="metric-label">CLASSIFIER</div>'
            f'<div class="metric-value">{"Ready" if present else "Files needed"}</div>'
            '<div class="metric-foot">Logistic Regression · TF-IDF</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="panel"><div class="metric-label">DATA SOURCE</div>'
            '<div class="metric-value">Kaggle</div>'
            '<div class="metric-foot">Latest dataset version on demand</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="panel"><div class="metric-label">FEATURES</div>'
            '<div class="metric-value">Text + Author</div>'
            '<div class="metric-foot">Inspect writing and label patterns</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Your workspace")
        st.markdown(
            '<div class="panel"><b>01 &nbsp; Model lab</b><br>'
            '<span class="metric-foot">Paste a passage to see the predicted author class and confidence.</span>'
            '<hr class="soft-divider"><b>02 &nbsp; Dataset explorer</b><br>'
            '<span class="metric-foot">Download the latest Kaggle data, filter it, and inspect label and text-length patterns.</span></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### Model files")
        for title, path in (
            ("Classifier", MODEL_PATH),
            ("TF-IDF vectorizer", VECTORIZER_PATH),
            ("Label encoder", ENCODER_PATH),
        ):
            state = "Available" if path.is_file() else "Not found"
            st.markdown(
                f'<div class="panel" style="padding:.8rem 1rem;margin-bottom:.55rem">'
                f'<span>{title}</span><span style="float:right;color:{"#72e1db" if path.is_file() else "#ffb4a6"}">{state}</span></div>',
                unsafe_allow_html=True,
            )


def render_model_lab() -> None:
    render_header(
        "Model lab",
        "Test a passage with the saved TF-IDF + Logistic Regression pipeline.",
        "Inference workspace",
    )
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown("### Your text")
        text = st.text_area(
            "Paste a sample to analyze",
            height=255,
            placeholder="Paste a paragraph here…",
            label_visibility="collapsed",
        )
        run = st.button("Analyze writing  →", type="primary", use_container_width=True)
        st.caption(f"{len(text.split())} words · {len(text)} characters")
    with right:
        st.markdown("### What you’ll get")
        st.markdown(
            '<div class="panel"><div class="eyebrow">MODEL OUTPUT</div>'
            '<p style="color:#c3c6dc;line-height:1.7">A predicted class and the model\'s probability distribution across its learned labels. Treat this as a model estimate, not proof of authorship.</p>'
            '<hr class="soft-divider"><span class="metric-foot">Short or out-of-domain text can reduce reliability.</span></div>',
            unsafe_allow_html=True,
        )

    if run:
        if not text.strip():
            st.warning("Enter some text before running an analysis.")
            return
        try:
            model, vectorizer, encoder = load_model_artifacts()
            features = vectorizer.transform([text])
            prediction = model.predict(features)[0]
            label = get_display_label(prediction, encoder)
            st.markdown("---")
            st.markdown("### Analysis")
            st.markdown(
                f'<div class="panel"><div class="result-label">Predicted author</div>'
                f'<div class="metric-value" style="font-size:2rem;margin:.3rem 0 1rem">{label}</div>'
                '<div class="metric-foot">Prediction from the saved project model</div></div>',
                unsafe_allow_html=True,
            )
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                class_labels = [
                    get_display_label(class_id, encoder) for class_id in model.classes_
                ]
                probability_frame = pd.DataFrame(
                    {"Author": class_labels, "Confidence": probabilities}
                ).sort_values("Confidence", ascending=False)
                best = float(probability_frame["Confidence"].max())
                st.metric("Top-class confidence", f"{best:.1%}")
                chart = px.bar(
                    probability_frame,
                    x="Confidence",
                    y="Author",
                    orientation="h",
                    range_x=[0, 1],
                    color="Confidence",
                    color_continuous_scale=["#72e1db", "#9b8cff"],
                )
                chart.update_layout(
                    height=240,
                    margin=dict(l=0, r=0, t=12, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c3c6dc",
                    coloraxis_showscale=False,
                    xaxis_tickformat=".0%",
                    xaxis_title="Model confidence",
                    yaxis_title=None,
                )
                st.plotly_chart(chart, use_container_width=True)
        except (FileNotFoundError, OSError, ValueError, AttributeError) as error:
            st.error(f"Could not run the model: {error}")


def render_dataset_explorer() -> None:
    render_header(
        "Dataset explorer",
        "Inspect the Kaggle collection of AI- and human-authored text.",
        "Data workspace",
    )
    if not st.session_state.get("dataset_requested", False):
        st.markdown(
            '<div class="panel"><h3>Bring the dataset into your workspace</h3>'
            '<p style="color:#aeb4cc">The latest Kaggle version is downloaded only when requested. '
            'Kaggle credentials may be required by kagglehub.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Load latest dataset  →", type="primary"):
            st.session_state["dataset_requested"] = True
            st.rerun()
        return

    try:
        data, source_files = load_dataset()
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not load the Kaggle dataset: {error}")
        st.caption("Check your internet connection and Kaggle API credentials, then try again.")
        if st.button("Try loading again"):
            load_dataset.clear()
            st.rerun()
        return

    text_column = find_column(data, ("Text", "text", "content", "article"))
    author_column = find_column(
        data, ("Author", "author", "label", "generated", "source", "type")
    )
    st.caption(f"Source files: {source_files}")
    if len(data) >= MAX_DATASET_ROWS:
        st.info(
            f"Showing up to the first {MAX_DATASET_ROWS:,} rows per CSV file for a responsive preview."
        )
    if text_column is None or author_column is None:
        st.warning(
            "Expected text and author/label columns were not both detected. "
            f"Available columns: {', '.join(map(str, data.columns))}"
        )
        st.dataframe(data.head(100), use_container_width=True, hide_index=True)
        return

    data = data.copy()
    data[text_column] = data[text_column].fillna("").astype(str)
    data[author_column] = data[author_column].fillna("Unknown").astype(str)
    data["Word count"] = data[text_column].str.split().str.len()

    metrics = st.columns(3)
    metrics[0].metric("Preview rows", f"{len(data):,}")
    metrics[1].metric("Author / label groups", f"{data[author_column].nunique():,}")
    metrics[2].metric("Average words", f"{data['Word count'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    label_filter, search_filter = st.columns([1, 2])
    labels = sorted(data[author_column].unique().tolist())
    with label_filter:
        selected_labels = st.multiselect(
            "Filter by author / label", labels, default=labels
        )
    with search_filter:
        search_text = st.text_input("Search text", placeholder="Find a phrase in a passage…")
    filtered = data[data[author_column].isin(selected_labels)]
    if search_text.strip():
        filtered = filtered[
            filtered[text_column].str.contains(search_text.strip(), case=False, regex=False)
        ]

    chart_col, length_col = st.columns(2, gap="large")
    with chart_col:
        st.markdown("### Author / label distribution")
        counts = filtered[author_column].value_counts().rename_axis("Author").reset_index(name="Rows")
        chart = px.bar(
            counts.head(15),
            x="Rows",
            y="Author",
            orientation="h",
            color="Rows",
            color_continuous_scale=["#72e1db", "#9b8cff"],
        )
        chart.update_layout(
            height=330,
            margin=dict(l=0, r=0, t=5, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c3c6dc",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending", "title": None},
            xaxis_title="Rows",
        )
        st.plotly_chart(chart, use_container_width=True)
    with length_col:
        st.markdown("### Passage length")
        chart = px.histogram(
            filtered,
            x="Word count",
            color=author_column,
            nbins=36,
            barmode="overlay",
            opacity=0.78,
            color_discrete_sequence=["#9b8cff", "#72e1db", "#ffc477", "#f28da8"],
        )
        chart.update_layout(
            height=330,
            margin=dict(l=0, r=0, t=5, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c3c6dc",
            xaxis_title="Words per passage",
            yaxis_title="Rows",
            legend_title_text="Author / label",
        )
        st.plotly_chart(chart, use_container_width=True)

    st.markdown(f"### Text samples <span style='color:#9ca3bd'>· {len(filtered):,} matching rows</span>", unsafe_allow_html=True)
    st.dataframe(
        filtered[[text_column, author_column, "Word count"]].head(100),
        use_container_width=True,
        hide_index=True,
        height=390,
    )
    st.download_button(
        "Download filtered rows as CSV",
        data=filtered.drop(columns=["Word count"]).to_csv(index=False).encode("utf-8"),
        file_name="textlens_filtered_dataset.csv",
        mime="text/csv",
    )


with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✦</span>textlens</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Choose a workspace",
        ("Overview", "Model lab", "Dataset explorer"),
        label_visibility="collapsed",
    )
    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-foot">AI & HUMAN TEXT<br><br>Classifier: Logistic Regression<br>Features: TF-IDF</div>',
        unsafe_allow_html=True,
    )

if page == "Overview":
    render_overview()
elif page == "Model lab":
    render_model_lab()
else:
    render_dataset_explorer()