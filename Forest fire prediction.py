# Forest Fire Prediction – Single-screen UI with background and big result text

import base64
import streamlit as st
import pandas as pd
import numpy as np

# Optional oversampler; the app still runs without imblearn
try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMB = True
except Exception:
    HAS_IMB = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------- Page setup ----------------
st.set_page_config(page_title="Forest Fire Prediction", layout="wide")

# Background image helper (CSS + base64)
def set_background(local_image_path: str):
    with open(local_image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    /* Make main container translucent for readability */
    .block-container {{
        background: rgba(0,0,0,0.35);
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 8px;
    }}
    /* Left input card styling */
    .input-card {{
        background: rgba(0,0,0,0.55);
        padding: 1.25rem;
        border-radius: 8px;
        color: #fff;
    }}
    .danger-text {{
        color: #ffffff;
        font-weight: 800;
        font-size: 44px;
        line-height: 1.15;
        text-shadow: 0 2px 4px rgba(0,0,0,0.6);
    }}
    .percent-text {{
        color: #ffffff;
        font-weight: 900;
        font-size: 52px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.6);
    }}
    label, .stNumberInput label, .stSelectbox label {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set your background image file (place it in the repo)
# e.g., images/bg.jpg
BACKGROUND_IMAGE = "images/forest_fire_bg.jpg"  # update path if needed
try:
    set_background(BACKGROUND_IMAGE)
except Exception:
    pass  # fail gracefully if image isn't present; UI still works

# ---------------- Data loading ----------------
DEFAULT_CSV = "forestfires(ISRO).csv"

@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

data_source = st.sidebar.selectbox("Choose data source", ["Repository file", "Upload CSV"])
if data_source == "Repository file":
    try:
        df_raw = load_csv(DEFAULT_CSV)
    except Exception as e:
        st.error("Could not read forestfires(ISRO).csv; upload a CSV from the sidebar.")
        up = st.sidebar.file_uploader("Upload forest fires CSV", type=["csv"])
        if up is None:
            st.stop()
        df_raw = pd.read_csv(up)
else:
    up = st.sidebar.file_uploader("Upload forest fires CSV", type=["csv"])
    if up is None:
        st.warning("Upload a CSV to continue.")
        st.stop()
    df_raw = pd.read_csv(up)

if "Status" not in df_raw.columns:
    st.error("Target column 'Status' not found in the dataset.")
    st.stop()

# ---------------- Cyclical encoding for month/day ----------------
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
    # drop raw and temp numerics if present
    df = df.drop(columns=[c for c in ["month","day","month_num","day_num"] if c in df.columns])
    return df

df = add_cyc_features(df_raw)

# ---------------- Train/test split & pipeline ----------------
# Use numeric columns only; target is 'Status'
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric_cols_all if c != "Status"]
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median(numeric_only=True))
y = df["Status"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

if HAS_IMB:
    ros = RandomOverSampler(sampling_strategy=0.125, random_state=42)
    X_train_os, y_train_os = ros.fit_resample(X_train, y_train)
else:
    X_train_os, y_train_os = X_train, y_train

scaler = MinMaxScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_os), columns=features, index=X_train_os.index)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=features, index=X_test.index)

# Model and threshold
st.sidebar.header("Model and threshold")
n_estimators = st.sidebar.slider("RandomForest n_estimators", 100, 800, 300, 50)
max_depth   = st.sidebar.slider("RandomForest max_depth",     2, 30, 10, 1)
threshold   = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

clf = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth,
    class_weight="balanced", random_state=42
)
clf.fit(X_train_sc, y_train_os)

# ---------------- Layout: two columns like the screenshot ----------------
left, right = st.columns([0.45, 0.55])

with left:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#fff; font-weight:800;'>Forest Fire Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#ddd;'>Predict the percentage of Forest‑Fire Occurrence</p>", unsafe_allow_html=True)

    # Build a one-row input using only three fields; fill the rest from training medians
    med = X_train.median(numeric_only=True)

    # Only show these three inputs; map to dataset names temp, wind, RH
    with st.form("predict_form", border=False):
        temp = st.number_input("Temperature", value=float(med.get("temp", 25.0)))
        wind = st.number_input("Wind", value=float(med.get("wind", 5.0)))
        rh   = st.number_input("Humidity", value=float(med.get("RH", 40.0)))
        do_predict = st.form_submit_button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.write("")  # spacing

# Validation AUCs only (no accuracy/F1 table)
proba_test = clf.predict_proba(X_test_sc)[:, 1]
st.markdown("<h3 style='color:#fff;'>Validation</h3>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#cfe3ff;'>ROC-AUC: {roc_auc_score(y_test, proba_test):.3f}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#cfe3ff;'>PR-AUC:  {average_precision_score(y_test, proba_test):.3f}</p>", unsafe_allow_html=True)

# ---------------- Prepare single-row prediction from 3 fields ----------------
if do_predict:
    # Start from medians for all features, then override temp/wind/RH with user input
    user_row = med.copy()
    if "temp" in user_row.index: user_row["temp"] = temp
    if "wind" in user_row.index: user_row["wind"] = wind
    if "RH"   in user_row.index: user_row["RH"]   = rh

    # If cyc features present but raw month/day missing (as in our pipeline), med already contains them
    user_df = pd.DataFrame([user_row])[features]
    user_sc = pd.DataFrame(scaler.transform(user_df), columns=features)

    # Scalar probability for class 1 (fire)
    p = float(clf.predict_proba(user_sc)[:, 1])  # convert to scalar to avoid TypeError
    pct = round(p * 100, 1)
    label = "in Danger" if p >= threshold else "Safe"

    # Right panel big text
    with right:
        st.markdown(
            f"<div class='danger-text'>Your Forest is {label}.</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='danger-text'>Percentage of fire occurring is</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='percent-text'>{pct:.1f}%</div>",
            unsafe_allow_html=True
        )
