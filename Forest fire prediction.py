# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve
)

from imblearn.over_sampling import RandomOverSampler

st.set_page_config(page_title="Forest Fires Prediction", layout="wide")
st.title("Forest Fires Prediction")

# -------------------
# 1) Load dataset
# -------------------
DATA_PATH = st.sidebar.text_input("CSV path", "forestfires(ISRO).csv")
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

try:
    forest_fires = load_data(DATA_PATH)
    st.write("Dataset shape:", forest_fires.shape)
    st.dataframe(forest_fires.head(), use_container_width=True)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if "Status" not in forest_fires.columns:
    st.error("Target column 'Status' not found in dataset.")
    st.stop()

# -------------------
# 2) Cyclical encoding for month/day
# -------------------
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
day_map = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}

def to_month_num(v):
    if pd.isna(v): return np.nan
    if isinstance(v, str):
        v2 = v.strip().lower()[:3]
        return month_map.get(v2, np.nan)
    try:
        return int(v)
    except:
        return np.nan

def to_day_num(v):
    if pd.isna(v): return np.nan
    if isinstance(v, str):
        v2 = v.strip().lower()[:3]
        return day_map.get(v2, np.nan)
    try:
        return int(v)
    except:
        return np.nan

def add_cyc_features(df):
    df = df.copy()
    if "month" in df.columns:
        df["month_num"] = df["month"].apply(to_month_num)
        df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    if "day" in df.columns:
        df["day_num"] = df["day"].apply(to_day_num)
        df["day_sin"] = np.sin(2 * np.pi * df["day_num"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_num"] / 7)
    # drop original and temps if present
    drop_cols = [c for c in ["month","day","month_num","day_num"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

forest_fires_proc = add_cyc_features(forest_fires)

# -------------------
# 3) Features/target and split (stratified)
# -------------------
features = [c for c in forest_fires_proc.columns if c != "Status"]
X = forest_fires_proc[features].copy()
y = forest_fires_proc["Status"].astype(int).copy()

# basic NA fill for safety
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# -------------------
# 4) Oversample TRAIN ONLY to avoid leakage
# -------------------
ros = RandomOverSampler(sampling_strategy=0.125, random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)

# -------------------
# 5) Scale numeric columns (fit on train only)
# -------------------
num_cols = X_train_over.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
X_train_scaled = X_train_over.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train_over[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# -------------------
# 6) Sidebar: model + hyperparameters + threshold
# -------------------
st.sidebar.header("Model & Threshold")

model_name = st.sidebar.selectbox(
    "Choose model",
    ["KNN", "SVM (RBF)", "Random Forest", "Logistic Regression"]
)

if model_name == "KNN":
    n_neighbors = st.sidebar.slider("n_neighbors", 1, 30, 5, 1)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
elif model_name == "SVM (RBF)":
    C = st.sidebar.slider("C", 0.1, 10.0, 1.0, 0.1)
    gamma_opt = st.sidebar.selectbox("gamma", ["scale", "auto"])
    model = SVC(kernel="rbf", C=C, gamma=gamma_opt, probability=True, class_weight="balanced", random_state=42)
elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 100, 800, 300, 50)
    max_depth = st.sidebar.slider("max_depth", 2, 30, 10, 1)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", random_state=42
    )
else:
    C = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.01)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=C, solver="lbfgs")

threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

# -------------------
# 7) Train and evaluate
# -------------------
model.fit(X_train_scaled, y_train_over)
proba_test = model.predict_proba(X_test_scaled)[:, 1]
pred_default = (proba_test >= 0.5).astype(int)
pred_thresh = (proba_test >= threshold).astype(int)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Test metrics (threshold=0.5)")
    st.write("ROC-AUC:", round(roc_auc_score(y_test, proba_test), 3))
    st.write("PR-AUC:", round(average_precision_score(y_test, proba_test), 3))
    st.text(classification_report(y_test, pred_default, digits=3))

with col2:
    st.subheader(f"Test metrics (threshold={threshold:.2f})")
    st.text(classification_report(y_test, pred_thresh, digits=3))

# -------------------
# 8) Sidebar input form for a new prediction
# -------------------
st.sidebar.header("Enter New Fire Conditions")

# Build inputs from original dataset so users provide month/day and numbers
orig_cols = forest_fires.columns.tolist()
numeric_input_cols = []
for c in orig_cols:
    if c in ["Status","month","day"]: 
        continue
    if pd.api.types.is_numeric_dtype(forest_fires[c]):
        numeric_input_cols.append(c)

month_choice = st.sidebar.selectbox(
    "month", list(month_map.keys()), index=0
)
day_choice = st.sidebar.selectbox(
    "day of week", list(day_map.keys()), index=0
)

input_dict = {"month": month_choice, "day": day_choice}
for c in numeric_input_cols:
    col_min = float(forest_fires[c].min())
    col_max = float(forest_fires[c].max())
    col_mean = float(forest_fires[c].mean())
    # choose float input if the column is float-like
    default_step = 0.1 if (forest_fires[c].dtype.kind in "fc") else 1.0
    val = st.sidebar.number_input(
        c, min_value=col_min, max_value=col_max, value=col_mean, step=default_step
    )
    input_dict[c] = val

# Build a one-row DataFrame, then apply the same cyc features and scaling
user_raw = pd.DataFrame([input_dict])
user_proc = add_cyc_features(user_raw)

# Align columns to training features
for col in features:
    if col not in user_proc.columns:
        user_proc[col] = 0.0
user_proc = user_proc[features].copy()
user_proc[num_cols] = scaler.transform(user_proc[num_cols])

# Predict
user_proba = model.predict_proba(user_proc)[:, 1]
user_label = int(user_proba >= threshold)
st.subheader("Single Prediction")
st.write(f"Predicted probability of fire (1): {user_proba:.3f}")
st.write(f"Predicted class at threshold {threshold:.2f}: {user_label} (1=Fire, 0=No Fire)")
