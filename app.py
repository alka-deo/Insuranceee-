import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import plotly.express as px

st.set_page_config(layout="wide", page_title="HR Retention Dashboard")

st.title("HR Retention / Attrition Dashboard")

st.markdown("""
This dashboard lets HR analyze attrition and train simple predictive models.
- **Drop nulls:** The app ignores (drops) null values as requested.
- **Files:** Put your CSV in the same folder as `app.py` named `HR_sample.csv` or upload from the sidebar.
""")

@st.cache_data
def load_default(path="HR_sample.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        return pd.DataFrame()

# Load default sample dataset shipped with the repo
df = load_default()

# Allow user upload (optional)
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Uploaded dataset loaded.")
    except Exception as e:
        st.sidebar.error("Unable to read uploaded file: " + str(e))

if df is None or df.shape[0] == 0:
    st.stop()

# Drop nulls globally (user requested)
df = df.dropna().copy()

# Sidebar: column mapping & filters
st.sidebar.header("Column mapping & filters")
job_col = st.sidebar.selectbox("Job Role column (for filter)", options=[None] + list(df.columns), index=1 if "JobRole" in df.columns else 0)
satisfaction_col = st.sidebar.selectbox("Satisfaction numeric column", options=[None] + list(df.columns), index=2 if "Satisfaction" in df.columns else 0)
target_col = st.sidebar.selectbox("Target (attrition) column", options=[None] + list(df.columns), index=7 if "Attrition" in df.columns else 0)
positive_label = st.sidebar.text_input("Positive label value (for target)", value="Yes")

st.sidebar.markdown("---")
st.sidebar.write("Chart filters")
# Role multiselect
roles = None
if job_col and job_col in df.columns:
    try:
        roles = st.sidebar.multiselect("Filter by Job Role (multi-select)", options=sorted(df[job_col].astype(str).unique()), default=sorted(df[job_col].astype(str).unique()))
    except Exception:
        roles = None
# Satisfaction slider
sat_slider = None
if satisfaction_col and satisfaction_col in df.columns and pd.api.types.is_numeric_dtype(df[satisfaction_col]):
    vmin = float(df[satisfaction_col].min())
    vmax = float(df[satisfaction_col].max())
    sat_slider = st.sidebar.slider("Satisfaction range", min_value=vmin, max_value=vmax, value=(vmin, vmax))

# Apply filters
df_filtered = df.copy()
if roles is not None and len(roles) > 0 and job_col:
    df_filtered = df_filtered[df_filtered[job_col].astype(str).isin(roles)].copy()
if sat_slider is not None and satisfaction_col in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered[satisfaction_col] >= sat_slider[0]) & (df_filtered[satisfaction_col] <= sat_slider[1])].copy()

# Tabs for Charts, Models, Predict
tab1, tab2, tab3 = st.tabs(["Charts", "Models (Train & Evaluate)", "Predict & Download"])

with tab1:
    st.header("Five actionable charts (with insights)")
    # 1) Attrition rate by job role (bar + sorting)
    st.subheader("1) Attrition rate by Job Role")
    if job_col and target_col and job_col in df_filtered.columns and target_col in df_filtered.columns:
        try:
            pos = str(positive_label)
            rates = df_filtered.groupby(job_col)[target_col].apply(lambda s: (s.astype(str) == pos).mean()).reset_index(name="attrition_rate")
            rates = rates.sort_values("attrition_rate", ascending=False)
            fig = px.bar(rates, x=job_col, y="attrition_rate", title=f"Attrition rate ({pos}) by {job_col}", text="attrition_rate")
            st.plotly_chart(fig, use_container_width=True)
            # Suggestion
            high = rates.iloc[0]
            st.info(f"Highest attrition: {high[{rates.columns[0]}]} — {high['attrition_rate']:.2%}")
        except Exception as e:
            st.info("Could not compute attrition rate by role: " + str(e))
    else:
        st.info("Select Job Role and Target columns in the sidebar to enable this chart.")

    # 2) Attrition vs Satisfaction (boxplot + mean diff)
    st.subheader("2) Attrition vs Satisfaction (boxplot & mean comparison)")
    if satisfaction_col and target_col and satisfaction_col in df_filtered.columns and target_col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[satisfaction_col]):
        try:
            df_tmp = df_filtered.copy()
            df_tmp["target_flag"] = (df_tmp[target_col].astype(str) == str(positive_label)).astype(int)
            fig2 = px.box(df_tmp, x="target_flag", y=satisfaction_col, points="all", labels={"target_flag":"Target (1=Positive)"})
            st.plotly_chart(fig2, use_container_width=True)
            means = df_tmp.groupby("target_flag")[satisfaction_col].mean().to_dict()
            st.write(f"Mean satisfaction (target=1): {means.get(1, np.nan):.2f}, (target=0): {means.get(0, np.nan):.2f}")
        except Exception as e:
            st.info("Could not create Attrition vs Satisfaction plot: " + str(e))
    else:
        st.info("Need numeric Satisfaction and Target columns for this chart.")

    # 3) Satisfaction distribution (histogram + skew/kurtosis)
    st.subheader("3) Satisfaction distribution + summary stats")
    if satisfaction_col and satisfaction_col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[satisfaction_col]):
        fig3 = px.histogram(df_filtered, x=satisfaction_col, nbins=30, marginal="box", title=f"Distribution of {satisfaction_col}")
        st.plotly_chart(fig3, use_container_width=True)
        st.write(df_filtered[satisfaction_col].describe())
    else:
        st.info("Select a numeric satisfaction column to view distribution.")

    # 4) Tenure (or Age) vs Attrition (binned) — if TenureMonths or Age exists; otherwise pick numeric
    st.subheader("4) Tenure/Age bins vs Attrition (cohort-like)")
    numeric_candidates = [c for c in df_filtered.select_dtypes(include=np.number).columns if c not in [satisfaction_col]]
    if len(numeric_candidates) > 0 and target_col in df_filtered.columns:
        col = None
        # prefer tenure-like
        for cand in numeric_candidates:
            if "tenure" in cand.lower() or "month" in cand.lower():
                col = cand
                break
        if col is None:
            # fallback to age if present
            for cand in numeric_candidates:
                if "age" in cand.lower():
                    col = cand
                    break
        if col is None:
            col = numeric_candidates[0]
        try:
            df_tmp = df_filtered.copy()
            df_tmp["bin"] = pd.qcut(df_tmp[col], q=4, duplicates="drop")
            rates = df_tmp.groupby("bin")[target_col].apply(lambda s: (s.astype(str) == str(positive_label)).mean()).reset_index(name="attrition_rate")
            fig4 = px.bar(rates, x="bin", y="attrition_rate", title=f"Attrition rate by {col} bins", text="attrition_rate")
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as e:
            st.info("Could not compute tenure/age bins chart: " + str(e))
    else:
        st.info("Not enough numeric columns and a target column to show this chart.")

    # 5) Correlation heatmap (numeric) — quick insight into drivers
    st.subheader("5) Correlation heatmap (numeric features)")
    num = df_filtered.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        try:
            corr = num.corr()
            fig5 = px.imshow(corr, text_auto=True, title="Correlation heatmap (numeric features)")
            st.plotly_chart(fig5, use_container_width=True)
        except Exception as e:
            st.info("Could not compute correlation heatmap: " + str(e))
    else:
        st.info("Not enough numeric features for a heatmap.")

with tab2:
    st.header("Train 3 models and display performance metrics")
    st.markdown("Models: Logistic Regression, Random Forest, Gradient Boosting. Choose target & features, then click Train.")
    with st.form("train_form"):
        target_train = st.selectbox("Target column", options=[None] + list(df.columns), index=1 if "Attrition" in df.columns else 0)
        pos_label = st.text_input("Positive label (for target)", value="Yes")
        feat_options = [c for c in df.columns if c != target_train]
        feats = st.multiselect("Feature columns (leave empty to use all except target)", options=feat_options)
        submit = st.form_submit_button("Train models")
    if submit:
        if target_train is None or target_train == "None":
            st.error("Select a target column to train models.")
        else:
            try:
                D = df.dropna(subset=[target_train]).copy()
                if feats:
                    X = D[feats].copy()
                else:
                    X = D.drop(columns=[target_train]).copy()
                y = (D[target_train].astype(str) == str(pos_label)).astype(int)
                # encode categoricals
                X_proc = X.copy()
                encoders = {}
                for c in X_proc.select_dtypes(include=['object','category']).columns.tolist():
                    le = LabelEncoder()
                    X_proc[c] = le.fit_transform(X_proc[c].astype(str))
                    encoders[c] = le
                numcols = X_proc.select_dtypes(include=np.number).columns.tolist()
                scaler = None
                if len(numcols) > 0:
                    scaler = StandardScaler()
                    X_proc[numcols] = scaler.fit_transform(X_proc[numcols])
                X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.25, random_state=42, stratify=y if len(set(y))>1 else None)

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(n_estimators=200),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200)
                }
                results = {}
                trained = {}
                for name, m in models.items():
                    try:
                        m.fit(X_train, y_train)
                        preds = m.predict(X_test)
                        probs = m.predict_proba(X_test)[:,1] if hasattr(m, "predict_proba") else None
                        acc = accuracy_score(y_test, preds)
                        roc = roc_auc_score(y_test, probs) if probs is not None and len(set(y_test))>1 else None
                        cr = classification_report(y_test, preds, output_dict=True, zero_division=0)
                        results[name] = {"accuracy": acc, "roc_auc": roc, "report": cr}
                        trained[name] = {"model": m, "encoders": encoders, "scaler": scaler, "features": X_proc.columns.tolist()}
                    except Exception as e:
                        results[name] = {"error": str(e)}
                st.success("Training finished.")
                for name, res in results.items():
                    st.subheader(name)
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        st.write(f"Accuracy: {res['accuracy']:.4f}")
                        st.write(f"ROC-AUC: {res['roc_auc'] if res['roc_auc'] is not None else 'N/A'}")
                        st.dataframe(pd.DataFrame(res["report"]).transpose())
                # feature importance for RF if available
                if "Random Forest" in trained:
                    try:
                        rf = trained["Random Forest"]["model"]
                        feats_rf = trained["Random Forest"]["features"]
                        importances = rf.feature_importances_
                        fi = pd.DataFrame({"feature": feats_rf, "importance": importances}).sort_values("importance", ascending=False).head(20)
                        st.subheader("Top feature importances (Random Forest)")
                        st.plotly_chart(px.bar(fi, x="feature", y="importance", title="Feature importances (RF)"), use_container_width=True)
                    except Exception as e:
                        st.info("Could not compute feature importances: " + str(e))
                # store trained models in session state for prediction tab
                st.session_state["trained_models"] = trained
                st.session_state["training_positive_label"] = pos_label
            except Exception as e:
                st.error("Training failed: " + str(e))

with tab3:
    st.header("Upload new dataset, predict attrition, and download results")
    uploaded2 = st.file_uploader("Upload CSV to predict on (optional)", type=["csv"], key="predict_upload")
    if uploaded2 is not None:
        try:
            df_new = pd.read_csv(uploaded2).dropna().copy()
            st.write("Preview of uploaded data (nulls dropped):")
            st.dataframe(df_new.head(20))
        except Exception as e:
            st.error("Unable to read uploaded file: " + str(e))
            df_new = None

        model_ready = False
        model = None
        encoders = {}
        scaler = None
        features = None
        # Prefer session-trained RF
        if "trained_models" in st.session_state and "Random Forest" in st.session_state["trained_models"]:
            model = st.session_state["trained_models"]["Random Forest"]["model"]
            encoders = st.session_state["trained_models"]["Random Forest"]["encoders"]
            scaler = st.session_state["trained_models"]["Random Forest"]["scaler"]
            features = st.session_state["trained_models"]["Random Forest"]["features"]
            model_ready = True
        else:
            # Try quick train on the cleaned df if a target column exists in the original upload mapping
            if target_col and target_col in df.columns:
                try:
                    D = df.dropna(subset=[target_col]).copy()
                    X = D.drop(columns=[target_col]).copy()
                    y = (D[target_col].astype(str) == str(positive_label)).astype(int)
                    X_proc = X.copy()
                    encoders = {}
                    for c in X_proc.select_dtypes(include=['object','category']).columns.tolist():
                        le = LabelEncoder()
                        X_proc[c] = le.fit_transform(X_proc[c].astype(str))
                        encoders[c] = le
                    numcols = X_proc.select_dtypes(include=np.number).columns.tolist()
                    if len(numcols) > 0:
                        scaler = StandardScaler()
                        X_proc[numcols] = scaler.fit_transform(X_proc[numcols])
                    rf = RandomForestClassifier(n_estimators=200)
                    rf.fit(X_proc, y)
                    model = rf
                    features = X_proc.columns.tolist()
                    model_ready = True
                    st.info("Trained a quick Random Forest on the current dataset for ad-hoc predictions.")
                except Exception as e:
                    st.info("Could not train quick model for predictions: " + str(e))

        if df_new is not None and model_ready:
            try:
                # keep common features
                common = [c for c in features if c in df_new.columns]
                X_new = df_new[common].copy()
                # encode categories using encoders if available
                for c in X_new.select_dtypes(include=['object','category']).columns.tolist():
                    if c in encoders:
                        try:
                            X_new[c] = encoders[c].transform(X_new[c].astype(str))
                        except Exception:
                            X_new[c] = LabelEncoder().fit_transform(X_new[c].astype(str))
                    else:
                        X_new[c] = LabelEncoder().fit_transform(X_new[c].astype(str))
                # scale numeric if scaler exists
                numcols = X_new.select_dtypes(include=np.number).columns.tolist()
                if scaler is not None and len(numcols) > 0:
                    try:
                        X_new[numcols] = scaler.transform(X_new[numcols])
                    except Exception:
                        pass
                preds = model.predict(X_new.fillna(0))
                df_new["predicted_"+str(target_col if target_col else "label")] = preds
                st.write("Predictions added:")
                st.dataframe(df_new.head(50))
                towrite = BytesIO()
                df_new.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button("Download predictions CSV", data=towrite, file_name="predictions_with_labels.csv", mime="text/csv")
            except Exception as e:
                st.error("Prediction failed: " + str(e))
        else:
            st.info("No model available. Train models in the Models tab or set a target column in the sidebar.")

st.markdown("---")
st.caption("Notes: The app drops nulls (as requested). Use the sidebar to map columns. Requirements file includes default packages without pinned versions.")