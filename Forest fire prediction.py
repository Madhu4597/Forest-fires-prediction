# ----------------------- app.py -----------------------
# Forest Fires Prediction (dynamic input -> predict fire / no fire)

import streamlit as st
import pandas as pd
import numpy as np

# Try to import optional packages; continue gracefully if missing
try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMB = True
except Exception:
    HAS_IMB = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

st.set_page_config(page_title="Forest Fires Prediction", layout="wide")
st.title("Forest Fires Prediction")

# ----------------------- Data loading -----------------------
DEFAULT_PATH = "forestfires(ISRO).csv"
path = st.sidebar.text_input("CSV path", value=DEFAULT_PATH)

@st.cache_data
def load_csv(p):
    df = pd.read_csv(p)
    return df

try:
    df_raw = load_csv(path)
except Exception as e:
    st.error(f"Could not read CSV at: {path}")
    st.exception(e)
    st.stop()

if "Status" not in df_raw.columns:
    st.error("Target column 'Status' is missing in the CSV.")
    st.stop()

# ----------------------- Cyclical encoding -----------------------
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
day_map = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}

def to_month_num(v):
    if pd.isna(v): return np.nan
    if isinstance(v, str):
        return month_map.get(v.strip().lower()[:3], np.nan)
    try:
        return int(v)
    except:
        return np.nan

def to_day_num(v):
    if pd.isna(v): return np.nan
    if isinstance(v, str):
        return day_map.get(v.strip().lower()[:3], np.nan)
    try:
        return int(v)
    except:
        return np.nan

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
    df = df.drop(columns=[c for c in ["month","day","month_num","day_num"] if c in df.columns])
    return df

df = add_cyc_features(df_raw)

# ----------------------- Train / Test split -----------------------
features = [c for c in df.columns if c != "Status"]
X = df[features].copy()
y = df["Status"].astype(int)

# basic numeric NA handling
X = X.replace([np.inf, -np.inf], np.nan)
if len(X.select_dtypes(include=[np.number]).columns) > 0:
    X = X.fillna(X.median(numeric_only=True))
else:
    X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ----------------------- Oversample TRAIN ONLY -----------------------
if HAS_IMB:
    ros = RandomOverSampler(sampling_strategy=0.125, random_state=42)
    X_train_os, y_train_os = ros.fit_resample(X_train, y_train)
else:
    X_train_os, y_train_os = X_train, y_train

# ----------------------- Scale numeric columns -----------------------
num_cols = X_train_os.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
X_train_sc = X_train_os.copy()
X_test_sc = X_test.copy()

if num_cols:
    X_train_sc[num_cols] = scaler.fit_transform(X_train_os[num_cols])
    X_test_sc[num_cols] = scaler.transform(X_test[num_cols])

# ----------------------- Sidebar model & threshold -----------------------
st.sidebar.header("Model and threshold")
model_name = st.sidebar.selectbox(
    "Choose model", ["Random Forest", "Logistic Regression", "KNN", "SVM (RBF)"]
)

if model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 100, 800, 300, 50)
    max_depth = st.sidebar.slider("max_depth", 2, 30, 10, 1)
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", random_state=42
    )
elif model_name == "Logistic Regression":
    C = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.01)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
elif model_name == "KNN":
    k = st.sidebar.slider("n_neighbors", 1, 30, 5, 1)
    model = KNeighborsClassifier(n_neighbors=k)
else:
    C = st.sidebar.slider("C", 0.1, 10.0, 1.0, 0.1)
    gamma = st.sidebar.selectbox("gamma", ["scale", "auto"])
    model = SVC(kernel="rbf", C=C, gamma=gamma, probability=True, class_weight="balanced", random_state=42)

threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

# ----------------------- Train & evaluate quickly -----------------------
model.fit(X_train_sc, y_train_os)
proba_test = model.predict_proba(X_test_sc)[:, 1]
pred = (proba_test >= threshold).astype(int)

colA, colB = st.columns(2)
with colA:
    st.subheader("Validation metrics")
    st.write("ROC-AUC:", round(roc_auc_score(y_test, proba_test), 3))
    st.write("PR-AUC:", round(average_precision_score(y_test, proba_test), 3))
with colB:
    st.text(classification_report(y_test, pred, digits=3))

# ----------------------- Dynamic user input -----------------------
st.sidebar.header("Enter new conditions")

# Re-create month/day inputs even though training used cyc features
months = list(month_map.keys())
days = list(day_map.keys())
month_choice = st.sidebar.selectbox("month", months, index=0)
day_choice = st.sidebar.selectbox("day of week", days, index=0)

# Use original numeric columns except Status/month/day
orig_numeric = []
for c in df_raw.columns:
    if c in ["Status","month","day"]:
        continue
    if pd.api.types.is_numeric_dtype(df_raw[c]):
        orig_numeric.append(c)

user = {"month": month_choice, "day": day_choice}
for c in orig_numeric:
    vmin = float(df_raw[c].min())
    vmax = float(df_raw[c].max())
    vmean = float(df_raw[c].mean())
    step = 0.1 if df_raw[c].dtype.kind in "fc" else 1.0
    user[c] = st.sidebar.number_input(c, min_value=vmin, max_value=vmax, value=vmean, step=step)

user_df = pd.DataFrame([user])
user_proc = add_cyc_features(user_df)

# align to training features
for col in features:
    if col not in user_proc.columns:
        user_proc[col] = 0.0
user_proc = user_proc[features].copy()
if num_cols:
    user_proc[num_cols] = scaler.transform(user_proc[num_cols])

# predict
user_proba = model.predict_proba(user_proc)[:, 1]
user_label = int(user_proba >= threshold)

st.subheader("Prediction for input")
st.write(f"Probability of fire (1): {user_proba:.3f}")
st.write(f"Predicted class at threshold {threshold:.2f}: {user_label}  (1=Fire, 0=No Fire)")
# ----------------------- end -----------------------
