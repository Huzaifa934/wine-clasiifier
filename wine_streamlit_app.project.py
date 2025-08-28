
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Wine Classifier Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍷 Wine Classifier — Interactive Dashboard")
st.caption("Sklearn Wine dataset • Train a model • Enter features • Get instant predictions")

# -----------------------
# Data Loading (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def load_data():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    target_names = data.target_names
    return X, y, target_names

X, y, target_names = load_data()

# Sidebar — model selection & training controls
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["Random Forest", "SVC (RBF)"],
    index=0
)

test_size = st.sidebar.slider("Test size (fraction for test set)", 0.1, 0.5, 0.2, 0.05, help="Portion of the data reserved for testing.")
random_state = st.sidebar.number_input("Random state (reproducibility)", min_value=0, value=42, step=1)

# Hyperparameters per model
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 10)
    max_depth = st.sidebar.slider("max_depth (0 = None)", 0, 30, 0, 1)
    rf_params = dict(n_estimators=n_estimators, random_state=random_state)
    if max_depth > 0:
        rf_params["max_depth"] = max_depth

    @st.cache_resource(show_spinner=False)
    def build_model_rf(rf_params, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return model, (X_train, X_test, y_train, y_test), acc

    model, splits, test_acc = build_model_rf(rf_params, test_size, random_state)


else:  # SVC
    C = st.sidebar.slider("C (regularization)", 0.01, 10.0, 1.0, 0.01)
    gamma = st.sidebar.selectbox("gamma", ["scale", "auto"], index=0)
    probability = True  # enable probability for predict_proba

    @st.cache_resource(show_spinner=False)
    def build_model_svc(C, gamma, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=C, gamma=gamma, probability=True))])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return pipe, (X_train, X_test, y_train, y_test), acc

    model, splits, test_acc = build_model_svc(C, gamma, test_size, random_state)

X_train, X_test, y_train, y_test = splits

# -----------------------
# Layout: Metrics + Optional Quick Peek
# -----------------------
left, right = st.columns(2)
with left:
    st.subheader("📊 Test Accuracy")
    st.metric("Accuracy", f"{test_acc*100:.2f}%")
    st.caption("Accuracy computed on a held-out test set using the chosen model and hyperparameters.")

with right:
    st.subheader("ℹ️ Dataset Quick Facts")
    st.write(f"Samples: **{len(X)}** | Features: **{X.shape[1]}** | Classes: **{len(target_names)}**")
    st.write("Classes:", {i: name for i, name in enumerate(target_names)})

# -----------------------
# Feature Input Form
# -----------------------
st.markdown("---")
st.header("🧪 Enter Features to Predict Wine Class")

# Determine slider ranges based on dataset min/max
feature_ranges = {}
for col in X.columns:
    col_min = float(X[col].min())
    col_max = float(X[col].max())
    # Add a small buffer to min/max for better slider UX
    rng = (round(col_min - 0.05*(col_max-col_min), 3), round(col_max + 0.05*(col_max-col_min), 3))
    feature_ranges[col] = rng

# Option to pre-fill inputs from a random test sample
col_a, col_b = st.columns([1, 2])
with col_a:
    use_example = st.checkbox("Use a random test example", value=False, help="Prefill sliders with a real sample from the test set.")

if use_example and len(X_test) > 0:
    # pick a random row from X_test
    example_idx = np.random.randint(0, len(X_test))
    example_row = X_test.iloc[example_idx]
    example_true = y_test.iloc[example_idx]
else:
    example_row = None
    example_true = None

# Build the input UI dynamically
user_values = {}
grid_cols = st.columns(3)
for i, feature in enumerate(X.columns):
    col_min, col_max = feature_ranges[feature]
    default_val = float(example_row[feature]) if example_row is not None else float(X[feature].median())
    with grid_cols[i % 3]:
        user_values[feature] = st.slider(
            label=feature,
            min_value=float(col_min),
            max_value=float(col_max),
            value=float(default_val),
            step=float((col_max - col_min) / 200 if (col_max - col_min) > 0 else 0.001),
            format="%.3f"
        )

# Predict button
st.markdown("")
predict_clicked = st.button("🔮 Predict Wine Class", type="primary", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame([user_values], columns=X.columns)
    try:
        y_pred = model.predict(input_df)[0]
        pred_label = target_names[y_pred]
        st.success(f"**Predicted class: {pred_label}**")

        # Probability (if available)
        proba_text = ""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            prob_table = pd.DataFrame({
                "class": target_names,
                "probability": probs
            }).sort_values("probability", ascending=False, ignore_index=True)
            st.subheader("Class Probabilities")
            st.dataframe(prob_table, use_container_width=True)
        else:
            st.info("This model does not support probability estimates.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

    if example_true is not None:
        st.caption(f"Note: The example you pre-filled had true class **{target_names[example_true]}**.")

# -----------------------
# Model Diagnostics
# -----------------------
st.markdown("---")
st.header("🧭 Model Diagnostics")

# Confusion matrix & classification report on test set
y_test_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()

c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=[f"true_{n}" for n in target_names], columns=[f"pred_{n}" for n in target_names]))

    # Feature importance for RF
    if model_choice == "Random Forest" and hasattr(model, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False, ignore_index=True)
        st.subheader("Feature Importances (RF)")
        st.dataframe(importances, use_container_width=True)

with c2:
    st.subheader("Classification Report (Test Set)")
    st.dataframe(report_df.style.format(precision=3), use_container_width=True)

st.markdown("---")
st.caption("Built with 🧠 scikit-learn & Streamlit • Dataset: UCI Wine (via sklearn)")
