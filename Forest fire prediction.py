import base64
import streamlit as st
import pandas as pd
import numpy as np

# Optional oversampler; app still runs if imblearn isn’t available
try:
    from imblearn.over_sampling import RandomOverSampler
    HAS_IMB = True
except Exception:
    HAS_IMB = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score

# -------------------- Page + CSS (hero header + button + background) --------------------
st.set_page_config(page_title="Forest Fire Prediction", layout="wide")

def set_bg(local_image_path: str):
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
            .hero {{
                text-align: center;
                margin-top: 0.5rem;
                margin-bottom: 1.25rem;
            }}
            .hero h1 {{
                color: #ff8c00;
                font-size: 56px;
                font-weight: 900;
                margin: 0.25rem 0 0.5rem 0;
            }}
            .hero p {{
                color: #222;
                font-size: 18px;
                margin: 0;
            }}
            .card {{
                background: rgba(255,255,255,0.92);
                padding: 1.25rem;
                border-radius: 10px;
                box-shadow: 0 6px 16px rgba(0,0,0,0.15);
            }}
            .cta-btn button {{
                background-color: #ff8c00 !important;
                color: #fff !important;
                font-weight: 800 !important;
                border-radius: 6px !important;
                height: 48px !important;
                width: 260px !important;
            }}
            .big-result {{
                font-weight: 900;
                font-size: 34px;
            }}
            .big-percent {{
                font-weight: 900;
                font-size: 44px;
                color: #ff8c00;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# Optional background image placed in repo (you can remove if not needed)
set_bg("images/forest_fire_bg.jpg")

# -------------------- Data load --------------------
DEFAULT_CSV = "forestfires(ISRO).csv"
st.sidebar.header("Data source")
src = st.sidebar.selectbox("Choose data source", ["Repository file", "Upload CSV"])

@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

if src == "Repository file":
    try:
        df_raw = load_csv(DEFAULT_CSV)
    except Exception as e:
        st.error("Could not read forestfires(ISRO).csv. Upload the CSV instead.")
        up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if up is None:
            st.stop()
        df_raw = pd.read_csv(up)
else:
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.warning("Upload a CSV to continue.")
        st.stop()
    df_raw = pd.read_csv(up)

if "Status" not in df_raw.columns:
    st.error("Target column 'Status' not found.")
    st.stop()

# -------------------- Cyclical encoding helpers (month/day) --------------------
month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
             'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
day_map   = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}

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
    # accept various casings
    cols = {c.lower(): c for c in df.columns}
    if 'month' in cols:
        c = cols['month']
        df['month_num'] = df[c].apply(to_month_num)
        df['month_sin'] = np.sin(2*np.pi*df['month_num']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month_num']/12)
    if 'day' in cols:
        c = cols['day']
        df['day_num'] = df[c].apply(to_day_num)
        df['day_sin'] = np.sin(2*np.pi*df['day_num']/7)
        df['day_cos'] = np.cos(2*np.pi*df['day_num']/7)
    # drop raw and temps if present
    for c in ['month','day','month_num','day_num']:
        if c in df.columns: df.drop(columns=c, inplace=True)
    return df

# Preprocess training data
df = add_cyc_features(df_raw)

# Separate features/target
y = df["Status"].astype(int)
X = df.drop(columns=["Status"])

# Identify numeric and categorical features (after cyc features)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# One-hot encode categoricals for training
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Split (stratified) and oversample train only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
if HAS_IMB:
    ros = RandomOverSampler(sampling_strategy=0.125, random_state=42)
    X_train_os, y_train_os = ros.fit_resample(X_train, y_train)
else:
    X_train_os, y_train_os = X_train, y_train

# Scale everything (tree doesn’t need it but harmless and consistent)
scaler = MinMaxScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_os), columns=X_train_os.columns, index=X_train_os.index)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns,      index=X_test.index)

# Model + threshold
st.sidebar.header("Model and threshold")
n_estimators = st.sidebar.slider("RandomForest n_estimators", 100, 800, 300, 50)
max_depth   = st.sidebar.slider("RandomForest max_depth",     2, 30, 10, 1)
threshold   = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

clf = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth,
    class_weight="balanced", random_state=42
)
clf.fit(X_train_sc, y_train_os)

# -------------------- HERO header --------------------
st.markdown(
    """
    <div class="hero">
      <h1>Forest Fire Prediction</h1>
      <p>Predict the probability of Forest‑Fire Occurrence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Dynamic inputs for ALL features (including x,y) --------------------
# Use 3 columns per row to reduce height
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("")

with st.form("all_features_form"):
    inputs = {}
    # Use medians/modes to propose defaults
    medians = df_raw.median(numeric_only=True)
    modes   = df_raw.mode(dropna=True).iloc if not df_raw.mode(dropna=True).empty else pd.Series(dtype=object)

    # Keep the original feature names (pre-cyc) for user inputs
    all_feature_cols = [c for c in df_raw.columns if c != "Status"]

    # arrange as chunks of three
    for i in range(0, len(all_feature_cols), 3):
        row = all_feature_cols[i:i+3]
        cols = st.columns(len(row))
        for c, col in zip(row, cols):
            with col:
                if df_raw[c].dtype.kind in "iufc":  # numeric
                    vmin = float(np.nanmin(df_raw[c].values)) if len(df_raw[c].dropna()) else 0.0
                    vmax = float(np.nanmax(df_raw[c].values)) if len(df_raw[c].dropna()) else 100.0
                    vdef = float(medians.get(c, df_raw[c].dropna().mean() if len(df_raw[c].dropna()) else 0.0))
                    step = 0.1 if df_raw[c].dtype.kind in "fc" else 1.0
                    inputs[c] = st.number_input(c, value=vdef, min_value=vmin, max_value=vmax, step=step)
                else:  # categorical / string
                    cats = sorted(df_raw[c].dropna().astype(str).unique().tolist())
                    vdef = str(modes.get(c, cats if cats else ""))
                    inputs[c] = st.selectbox(c, cats if cats else [vdef], index=(cats.index(vdef) if vdef in cats else 0))

    submit = st.form_submit_button("PREDICT PROBABILITY", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Build a single-row input consistent with training --------------------
def build_user_row(raw_inputs: dict) -> pd.DataFrame:
    # Start from raw inputs in original schema, then apply cyc features
    user_df = pd.DataFrame([raw_inputs])

    # Ensure numeric types for numeric columns
    for c in user_df.columns:
        if c in df_raw.columns and df_raw[c].dtype.kind in "iufc":
            user_df[c] = pd.to_numeric(user_df[c], errors="coerce")

    # Apply cyclical encoding and drop month/day like training
    user_df = add_cyc_features(user_df)

    # One-hot encode exactly like training and align columns
    user_proc = pd.get_dummies(user_df, columns=[col for col in user_df.columns if user_df[col].dtype == 'object'], drop_first=True)
    user_proc = user_proc.reindex(columns=X.columns, fill_value=0.0)
    return user_proc

# -------------------- Validation (AUCs only) --------------------
proba_test = clf.predict_proba(X_test_sc)[:, 1]
st.write(f"ROC-AUC: {roc_auc_score(y_test, proba_test):.3f}")
st.write(f"PR-AUC:  {average_precision_score(y_test, proba_test):.3f}")

# -------------------- Predict and show big message --------------------
if submit:
    user_row = build_user_row(inputs)
    user_sc  = pd.DataFrame(scaler.transform(user_row), columns=user_row.columns)
    p = float(clf.predict_proba(user_sc)[:, 1])  # scalar probability
    pct = round(p * 100, 2)
    label = "in Danger" if p >= threshold else "safe"

    st.markdown(
        f"""
        <div style="margin-top: 1.25rem; text-align: center;">
            <div class="big-result">Your Forest is <b>{label}</b>.</div>
            <div class="big-result">Probability of fire occurring is</div>
            <div class="big-percent">{pct:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
