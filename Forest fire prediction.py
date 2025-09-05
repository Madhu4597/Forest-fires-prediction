# Forest Fire Prediction — Left: all CSV characteristics, Right: big result

import base64
import streamlit as st
import pandas as pd
import numpy as np

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

st.set_page_config(page_title="Forest Fire Prediction", layout="wide")

# ---------- Background helper (optional) ----------
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
            .title {{ font-size: 48px; font-weight: 900; color: #ff8c00; text-align:center; }}
            .subtitle {{ text-align:center; color:#222; margin-bottom:1rem; }}
            .result-big {{ font-weight: 900; font-size: 34px; color:#111; text-align:center; }}
            .result-pct {{ font-weight: 900; font-size: 52px; color:#ff8c00; text-align:center; }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

set_background("images/forest_fire_bg.jpg")  # adjust or remove

# ---------- Load CSV ----------
st.sidebar.header("Data source")
choice = st.sidebar.selectbox("Choose data source", ["Repository file", "Upload CSV"])
DEFAULT_CSV = "forestfires(ISRO).csv"

@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

if choice == "Repository file":
    try:
        df_raw = load_csv(DEFAULT_CSV)
    except Exception:
        up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if up is None: st.stop()
        df_raw = pd.read_csv(up)
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is None: st.stop()
    df_raw = pd.read_csv(up)

if "Status" not in df_raw.columns:
    st.error("Target column 'Status' missing in CSV.")
    st.stop()

# ---------- Cyclical helpers (internal only) ----------
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

# Train on engineered set; UI remains original columns
df_train = add_cyc_features(df_raw)
X_full = df_train.drop(columns=["Status"])
y = df_raw["Status"].astype(int)

num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_full.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

clf = RandomForestClassifier(
    n_estimators=st.sidebar.slider("RandomForest n_estimators", 100, 800, 400, 50),
    max_depth=st.sidebar.slider("RandomForest max_depth", 2, 30, 10, 1),
    class_weight="balanced", random_state=42
)
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.25, 0.01)

# Split and fit (oversample train only)
X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)

X_tr_tr = pre.fit_transform(X_tr)
X_te_tr = pre.transform(X_te)

if HAS_IMB:
    ros = RandomOverSampler(random_state=42)
    X_tr_tr, y_tr = ros.fit_resample(X_tr_tr, y_tr)

clf.fit(X_tr_tr, y_tr)

# ---------- Header ----------
st.markdown("<div class='title'>Forest Fire Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict the probability of Forest‑Fire Occurrence</div>", unsafe_allow_html=True)

# ---------- Two columns: left inputs, right result ----------
left, right = st.columns([0.55, 0.45], vertical_alignment="top")
with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Enter characteristics")

    # Build UI ONLY from original CSV columns (avoid engineered names)
    orig_cols = [c for c in df_raw.columns if c != "Status"]
    ui_values = {}

    # Lay out inputs in a grid of three columns
    cols = st.columns(3)
    for i, col in enumerate(orig_cols):
        with cols[i % 3]:
            s = df_raw[col]
            if col == "month":
                ui_values[col] = st.selectbox("month", list(month_map.keys()))
            elif col == "day":
                ui_values[col] = st.selectbox("day", list(day_map.keys()))
            else:
                if pd.api.types.is_numeric_dtype(s):
                    # Use df_raw for bounds to avoid KeyError on engineered names
                    numeric_s = pd.to_numeric(s, errors="coerce")
                    vmin = float(np.nanmin(numeric_s))
                    vmax = float(np.nanmax(numeric_s))
                    vmean = float(np.nanmean(numeric_s))
                    step = 0.1 if s.dtype.kind in "fc" else 1.0
                    ui_values[col] = st.number_input(col, value=vmean, min_value=vmin, max_value=vmax, step=step)
                else:
                    opts = sorted(list(s.dropna().astype(str).unique()))
                    ui_values[col] = st.selectbox(col, opts) if opts else st.text_input(col, "")

    predict_btn = st.button("Predict", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Prediction")
    # Quick validation (AUCs only)
    proba_val = clf.predict_proba(X_te_tr)[:, 1]
    st.write(f"ROC-AUC: {roc_auc_score(y_te, proba_val):.3f}")
    st.write(f"PR-AUC:  {average_precision_score(y_te, proba_val):.3f}")

    if predict_btn:
        # Create single-row original record, then add cyc features
        user_df = pd.DataFrame([ui_values])
        user_df_eng = add_cyc_features(user_df)

        # Align to training columns
        for c in X_full.columns:
            if c not in user_df_eng.columns:
                user_df_eng[c] = np.nan
        user_df_eng = user_df_eng[X_full.columns]

        # Transform and predict
        user_tr = pre.transform(user_df_eng)
        p = float(clf.predict_proba(user_tr)[:, 1])
        y_hat = int(p >= threshold)
        pct = p * 100.0
        label = "in Danger" if y_hat == 1 else "safe"

        st.markdown(f"<div class='result-big'>Your Forest is {label}.</div>", unsafe_allow_html=True)
        st.markdown("<div class='result-big'>Probability of fire occurring is</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-pct'>{pct:.2f}%</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
