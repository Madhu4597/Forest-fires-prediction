# Forest Fire Prediction — Best‑practice Streamlit app
# - UCI features including FWI components (FFMC, DMC, DC, ISI)
# - ROC‑AUC and PR‑AUC evaluation
# - Threshold slider to tune risk
# - Stratified split + train‑only oversampling
# - Left inputs, right prediction layout

import streamlit as st
import pandas as pd
import numpy as np
import base64

# Optional: imbalanced-learn for train-only oversampling
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
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------- Page & simple background (optional) ----------------
st.set_page_config(page_title="Forest Fire Prediction", layout="wide")

def set_background(local_image_path: str):
    try:
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
            .panel {{ background: rgba(255,255,255,0.92); padding: 1.1rem; border-radius: 10px; }}
            .title {{ font-size: 44px; font-weight: 900; color: #ff8c00; }}
            .result-big {{ font-weight: 900; font-size: 34px; color:#111; }}
            .result-pct {{ font-weight: 900; font-size: 52px; color:#ff8c00; }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# Adjust or remove this image path
set_background("images/forest_fire_bg.jpg")

# ---------------- Data loading ----------------
st.sidebar.header("Data source")
data_src = st.sidebar.selectbox("Choose data source", ["Repository file", "Upload CSV"])
DEFAULT_CSV = "forestfires(ISRO).csv"

@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

if data_src == "Repository file":
    try:
        df_raw = load_csv(DEFAULT_CSV)
    except Exception:
        upl = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if upl is None:
            st.error("Add forestfires(ISRO).csv to the repo or upload a CSV.")
            st.stop()
        df_raw = pd.read_csv(upl)
else:
    upl = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if upl is None:
        st.warning("Upload a CSV to continue.")
        st.stop()
    df_raw = pd.read_csv(upl)

if "Status" not in df_raw.columns:
    st.error("Target column 'Status' not found in the dataset.")
    st.stop()

# ---------------- Cyclical features for month/day (internal) ----------------
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

# Engineer features for model training; UI remains original CSV columns
df_train = add_cyc_features(df_raw)

# ---------------- Split features/target ----------------
target_col = "Status"
X_full = df_train.drop(columns=[target_col])
y = df_raw[target_col].astype(int)

num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_full.select_dtypes(exclude=[np.number]).columns.tolist()

# ---------------- Sidebar model & threshold ----------------
st.sidebar.header("Model & threshold")
n_estimators = st.sidebar.slider("RandomForest n_estimators", 100, 800, 400, 50)
max_depth   = st.sidebar.slider("RandomForest max_depth", 2, 30, 10, 1)
threshold   = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.25, 0.01)

# ---------------- Preprocess & fit with stratify + train-only oversampling ----------------
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

clf = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth,
    class_weight="balanced", random_state=42
)

X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.20, stratify=y, random_state=42)
X_tr_tr = pre.fit_transform(X_tr)
X_te_tr = pre.transform(X_te)

if HAS_IMB:
    ros = RandomOverSampler(random_state=42)
    X_tr_tr, y_tr = ros.fit_resample(X_tr_tr, y_tr)

clf.fit(X_tr_tr, y_tr)

# ---------------- Header ----------------
st.markdown("<div class='title'>Forest Fire Prediction</div>", unsafe_allow_html=True)
st.caption("Predict the probability of Forest‑Fire Occurrence")

# ---------------- Two-column layout ----------------
left, right = st.columns([0.6, 0.4], vertical_alignment="top")

with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Enter characteristics")

    # Form inputs from original CSV columns only (no engineered names)
    orig_cols = [c for c in df_raw.columns if c != "Status"]
    ui = {}
    grid = st.columns(3)
    for i, col in enumerate(orig_cols):
        with grid[i % 3]:
            s = df_raw[col]
            if col == "month":
                ui[col] = st.selectbox("month", list(month_map.keys()))
            elif col == "day":
                ui[col] = st.selectbox("day", list(day_map.keys()))
            else:
                if pd.api.types.is_numeric_dtype(s):
                    numeric_s = pd.to_numeric(s, errors="coerce")
                    vmin = float(np.nanmin(numeric_s))
                    vmax = float(np.nanmax(numeric_s))
                    vmean = float(np.nanmean(numeric_s))
                    step = 0.1 if s.dtype.kind in "fc" else 1.0
                    ui[col] = st.number_input(col, value=vmean, min_value=vmin, max_value=vmax, step=step)
                else:
                    opts = sorted(list(s.dropna().astype(str).unique()))
                    ui[col] = st.selectbox(col, opts) if opts else st.text_input(col, "")
    predict_btn = st.button("Predict", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Prediction")

    # Threshold‑independent evaluation
    proba_val = clf.predict_proba(X_te_tr)[:, 1]
    st.write(f"ROC-AUC: {roc_auc_score(y_te, proba_val):.3f}")
    st.write(f"PR-AUC:  {average_precision_score(y_te, proba_val):.3f}")

    if predict_btn:
        # Build one-row from UI, add cyc features, align to training columns
        user_df = pd.DataFrame([ui])
        user_eng = add_cyc_features(user_df)
        for c in X_full.columns:
            if c not in user_eng.columns:
                user_eng[c] = np.nan
        user_eng = user_eng[X_full.columns]

        user_tr = pre.transform(user_eng)
        p = float(clf.predict_proba(user_tr)[:, 1])
        pct = p * 100.0
        label = "in Danger" if p >= threshold else "safe"

        st.markdown(f"<div class='result-big'>Your Forest is {label}.</div>", unsafe_allow_html=True)
        st.markdown("<div class='result-big'>Probability of fire occurring is</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-pct'>{pct:.2f}%</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
