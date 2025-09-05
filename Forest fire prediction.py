# app.py — Forest Fire Prediction with full-feature form and styled UI

import base64
import streamlit as st
import pandas as pd
import numpy as np

# Optional oversampler; app still runs without it
try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMB = True
except Exception:
    HAS_IMB = False

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page and background ----------------
st.set_page_config(page_title="Forest Fire Prediction", layout="wide")

def set_background(local_image_path: str):
    with open(local_image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .hero {{
            text-align: center;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .title {{
            font-size: 56px;
            font-weight: 900;
            color: #ff9d00;
            text-shadow: 0 2px 4px rgba(0,0,0,0.35);
        }}
        .subtitle {{
            font-size: 18px;
            color: #222;
        }}
        .panel {{
            background: rgba(255,255,255,0.90);
            padding: 1.25rem;
            border-radius: 10px;
        }}
        .result-big {{
            font-weight: 900;
            font-size: 36px;
            color: #111;
        }}
        .result-pct {{
            font-weight: 900;
            font-size: 54px;
            color: #ff9d00;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Update this path to a local image in your repo (or leave try/except)
BACKGROUND_IMAGE = "images/forest_fire_bg.jpg"
try:
    set_background(BACKGROUND_IMAGE)
except Exception:
    pass

# ---------------- Data loading ----------------
st.sidebar.header("Data source")
source = st.sidebar.selectbox("Choose data source", ["Repository file", "Upload CSV"])

DEFAULT_CSV = "forestfires(ISRO).csv"

@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

if source == "Repository file":
    try:
        df_raw = load_csv(DEFAULT_CSV)
    except Exception:
        upl = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if upl is None:
            st.error("Add forestfires(ISRO).csv to the repo or upload a CSV from the sidebar.")
            st.stop()
        df_raw = pd.read_csv(upl)
else:
    upl = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if upl is None:
        st.warning("Upload a CSV to continue.")
        st.stop()
    df_raw = pd.read_csv(upl)

if "Status" not in df_raw.columns:
    st.error("Target column 'Status' is missing.")
    st.stop()

# ---------------- Cyclical encoding helpers ----------------
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
day_map = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}

def to_month_num(v):
    if pd.isna(v): return np.nan
    if isinstance(v, str): return month_map.get(v.strip().lower()[:3], np.nan)
    try: return int(v)
    except: return np.nan

def to_day_num(v):
    if pd.isna(v): return np.nan
    if isinstance(v, str): return day_map.get(v.strip().lower()[:3], np.nan)
    try: return int(v)
    except: return np.nan

def add_cyc_features(df):
    df = df.copy()
    if "month" in df.columns:
        df["month_num"] = df["month"].apply(to_month_num)
        df["month_sin"] = np.sin(2*np.pi*df["month_num"]/12)
        df["month_cos"] = np.cos(2*np.pi*df["month_num"]/12)
    if "day" in df.columns:
        df["day_num"] = df["day"].apply(to_day_num)
        df["day_sin"] = np.sin(2*np.pi*df["day_num"]/7)
        df["day_cos"] = np.cos(2*np.pi*df["day_num"]/7)
    return df

df_base = add_cyc_features(df_raw)

# ---------------- Hero ----------------
st.markdown(
    "<div class='hero'><div class='title'>Forest Fire Prediction</div>"
    "<div class='subtitle'>Predict the probability of Forest‑Fire Occurrence</div></div>",
    unsafe_allow_html=True,
)

# ---------------- Feature/target split ----------------
target_col = "Status"
X_full = df_base.drop(columns=[target_col])
y = df_raw[target_col].astype(int)

# Identify columns
num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_full.select_dtypes(exclude=[np.number]).columns.tolist()

# ---------------- Sidebar model/threshold ----------------
st.sidebar.header("Model & threshold")
n_estimators = st.sidebar.slider("RandomForest n_estimators", 100, 800, 300, 50)
max_depth   = st.sidebar.slider("RandomForest max_depth", 2, 30, 10, 1)
threshold   = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

# ---------------- Build preprocessing + model ----------------
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", MinMaxScaler()),
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
], remainder="drop")

clf = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth,
    class_weight="balanced", random_state=42
)

# Fit/transform safely with stratified split and train-only resampling
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)

if HAS_IMB:
    # Fit the preprocessor on X_train, then oversample in transformed space for robustness
    X_train_tr = pre.fit_transform(X_train)
    ros = RandomOverSampler(random_state=42)
    X_train_os, y_train_os = ros.fit_resample(X_train_tr, y_train)
    # Train model on oversampled transformed data
    clf.fit(X_train_os, y_train_os)
else:
    # Simple pipeline when imblearn isn't available
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)

# ---------------- Dynamic full-feature form ----------------
st.markdown("<div class='panel'>", unsafe_allow_html=True)

# Build widgets for all feature columns in a 3-column grid
cols = st.columns(3)
user_row = {}

for i, col in enumerate(X_full.columns):
    with cols[i % 3]:
        if col == "month":
            user_row[col] = st.selectbox("month", list(month_map.keys()))
        elif col == "day":
            user_row[col] = st.selectbox("day", list(day_map.keys()))
        else:
            if pd.api.types.is_numeric_dtype(X_full[col]):
                vmin = float(np.nanmin(pd.to_numeric(df_raw[col], errors="coerce")))
                vmax = float(np.nanmax(pd.to_numeric(df_raw[col], errors="coerce")))
                vmean = float(np.nanmean(pd.to_numeric(df_raw[col], errors="coerce")))
                step = 0.1 if df_raw[col].dtype.kind in "fc" else 1.0
                user_row[col] = st.number_input(col, value=vmean, min_value=vmin, max_value=vmax, step=step)
            else:
                # categorical
                opts = sorted(list(df_raw[col].dropna().astype(str).unique()))
                user_row[col] = st.selectbox(col, opts) if opts else st.text_input(col, "")

predict_btn = st.button("PREDICT PROBABILITY", type="primary")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction and big text ----------------
if predict_btn:
    user_df = pd.DataFrame([user_row])

    # Add cyclical features (month/day) into the single row too
    user_df = add_cyc_features(user_df)

    # Align columns to training X_full columns (order matters for transformers)
    for c in X_full.columns:
        if c not in user_df.columns:
            user_df[c] = np.nan
    user_df = user_df[X_full.columns]

    # Predict probability of class 1 (fire)
    if HAS_IMB:
        # Use the same preprocessor learned on X_train
        user_tr = pre.transform(user_df)
        proba = float(clf.predict_proba(user_tr)[:, 1])
    else:
        proba = float(model.predict_proba(user_df)[:, 1])

    pct = proba * 100.0
    label = "in Danger" if proba >= threshold else "safe"

    st.markdown(
        f"<div class='panel' style='text-align:center;'>"
        f"<div class='result-big'>Your Forest is {label.capitalize()}.</div>"
        f"<div class='result-big'>Probability of fire occurring is</div>"
        f"<div class='result-pct'>{pct:.2f}%</div>"
        f"</div>",
        unsafe_allow_html=True
    )
