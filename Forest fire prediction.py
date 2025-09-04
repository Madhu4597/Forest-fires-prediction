# ----------------------- app.py -----------------------
# Forest Fires Prediction (dynamic input -> predict fire / no fire)

import streamlit as st
import pandas as pd
import numpy as np

# Optional oversampler; app still runs if imblearn is unavailable
try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMB = True
except Exception:
    HAS_IMB = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

st.set_page_config(page_title="https://github.com/Madhu4597/Forest-fires-prediction/blob/main/forestfires(ISRO).csv", layout="wide")
st.title("Forest Fires Prediction")

# ---------- Load data (repo file or upload) ----------
DEFAULT_CSV = "forestfires(ISRO).csv"
st.sidebar.header("Data source")
src = st.sidebar.selectbox("Choose data source", ["Repository file", "Upload CSV"])

@st.cache_data
def load_csv_repo(path: str):
    return pd.read_csv(path)

def load_csv_safe():
    if src == "Repository file":
        return load_csv_repo(DEFAULT_CSV)
    up = st.sidebar.file_uploader("Upload forest fires CSV", type=["csv"])
    if up is None:
        st.warning("Upload a CSV or switch to 'Repository file'.")
        st.stop()
    return pd.read_csv(up)

try:
    df_raw = load_csv_safe()
except Exception as e:
    st.error("Failed to load CSV; check filename/location or upload a file.")
    st.exception(e)
    st.stop()

if "Status" not in df_raw.columns:
    st.error("Target column 'Status' not found in the dataset.")
    st.stop()

# ---------- Cyclical encoding for month/day ----------
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
    # drop original string/temporary columns if present
    df = df.drop(columns=[c for c in ["month","day","month_num","day_num"] if c in df.columns])
    return df

df = add_cyc_features(df_raw)

# ---------- Select features and split (stratified) ----------
# Keep only numeric columns for modeling
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in all_numeric if c != "Status"]

X = df[features].copy()
y = df["Status"].astype(int)

# Numeric NA handling
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ---------- Oversample TRAIN ONLY (optional) ----------
if HAS_IMB:
    ros = RandomOverSampler(sampling_strategy=0.125, random_state=42)
    X_train_os, y_train_os = ros.fit_resample(X_train, y_train)
else:
    X_train_os, y_train_os = X_train, y_train

# ---------- Scale numerics ----------
num_cols = X_train_os.columns.tolist()
scaler = MinMaxScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_os[num_cols]), columns=num_cols, index=X_train_os.index)
X_test_sc  = pd.DataFrame(scaler.transform(X_test[num_cols]),       columns=num_cols, index=X_test.index)

# ---------- Train a robust default model ----------
st.sidebar.header("Model and threshold")
n_estimators = st.sidebar.slider("RandomForest n_estimators", 100, 800, 300, 50)
max_depth   = st.sidebar.slider("RandomForest max_depth",     2, 30, 10, 1)
threshold   = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

clf = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth,
    class_weight="balanced", random_state=42
)
clf.fit(X_train_sc, y_train_os)

# ---------- Quick validation metrics ----------
proba = clf.predict_proba(X_test_sc)[:, 1]
pred  = (proba >= threshold).astype(int)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Validation")
    st.write("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))
    st.write("PR-AUC:",  round(average_precision_score(y_test, proba), 3))
with c2:
    st.text(classification_report(y_test, pred, digits=3))

# ---------- Dynamic user input ----------
st.sidebar.header("Enter new conditions")

# month/day selectors (original dataset style for user)
months = list(month_map.keys())
days   = list(day_map.keys())
month_choice = st.sidebar.selectbox("month", months, index=0)
day_choice   = st.sidebar.selectbox("day of week", days, index=0)

# numeric inputs from original dataset (excluding month/day/Status)
numeric_cols_original = [
    c for c in df_raw.columns
    if c not in ["Status","month","day"] and pd.api.types.is_numeric_dtype(df_raw[c])
]

user_in = {"month": month_choice, "day": day_choice}
for c in numeric_cols_original:
    vmin, vmax, vmean = float(df_raw[c].min()), float(df_raw[c].max()), float(df_raw[c].mean())
    step = 0.1 if df_raw[c].dtype.kind in "fc" else 1.0
    user_in[c] = st.sidebar.number_input(c, min_value=vmin, max_value=vmax, value=vmean, step=step)

# build one-row frame and apply same preprocessing
user_df = pd.DataFrame([user_in])
user_proc = add_cyc_features(user_df)

# align to training features
for col in features:
    if col not in user_proc.columns:
        user_proc[col] = 0.0
user_proc = user_proc[features].copy()

# scale with train scaler
user_proc[num_cols] = scaler.transform(user_proc[num_cols])

# predict
user_p = clf.predict_proba(user_proc)[:, 1]
user_y = int(user_p >= threshold)

st.subheader("Prediction for input")
st.write(f"Probability of fire (1): {user_p:.3f}")
st.write(f"Predicted class at threshold {threshold:.2f}: {user_y} (1=Fire, 0=No Fire)")
# ----------------------- end -----------------------

