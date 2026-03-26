import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="ProphetPrice | Analytics", layout="wide", page_icon="🏠")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #1f77b4; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)


# ================================
# DATA LOADING & PRE-PROCESSING
# ================================
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv("house_price_dataset.csv")
    except FileNotFoundError:
        # Fallback synthetic data for demonstration
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'Area_sqft': np.random.randint(500, 5000, n),
            'Bedrooms': np.random.randint(1, 6, n),
            'Bathrooms': np.random.randint(1, 4, n),
            'Floors': np.random.randint(1, 4, n),
            'Age_Years': np.random.randint(0, 50, n),
            'Location_Score': np.random.randint(1, 11, n),
            'Distance_City_km': np.random.randint(1, 50, n),
            'Price_INR': np.random.randint(2000000, 20000000, n)
        })

    # Identify target and features dynamically
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1].tolist()
    return df, feature_cols, target_col


df, features, target = load_and_prep_data()


# ================================
# MODEL TRAINING ENGINE
# ================================
@st.cache_resource
def train_suite(data, feature_names, target_name):
    X = data[feature_names].values
    y = data[target_name].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sc_X = StandardScaler()
    sc_y = StandardScaler()

    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.transform(X_test)
    y_train_scaled = sc_y.fit_transform(y_train)

    models = {
        "Linear Regression": LinearRegression(),
        "SVR (RBF)": SVR(kernel='rbf'),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    metrics = {}
    model_objects = {}
    preds_store = {}

    for name, model in models.items():
        # SVR needs flat y array
        if name == "SVR (RBF)":
            model.fit(X_train_scaled, y_train_scaled.ravel())
        else:
            model.fit(X_train_scaled, y_train_scaled)

        y_pred_scaled = model.predict(X_test_scaled).reshape(-1, 1)
        y_pred = sc_y.inverse_transform(y_pred_scaled)

        metrics[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred)
        }
        model_objects[name] = model
        preds_store[name] = y_pred

    return metrics, model_objects, sc_X, sc_y, y_test, preds_store


results, trained_models, scaler_X, scaler_y, y_actual, all_preds = train_suite(df, features, target)

# ================================
# SIDEBAR CONTROLS
# ================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/609/609036.png", width=100)
st.sidebar.title("Property Config")

user_inputs = []
for col in features:
    val = st.sidebar.number_input(f"Enter {col}", value=int(df[col].median()))
    user_inputs.append(val)

predict_btn = st.sidebar.button("Generate Valuation")

# ================================
# MAIN DASHBOARD LAYOUT
# ================================
st.title("🏠 Real Estate Intelligence Dashboard")
st.markdown(f"**Target Variable:** `{target}` | **Features Analyzed:** `{', '.join(features)}`")

# 1. TOP METRICS
best_model = max(results, key=lambda x: results[x]['R2'])
m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset Size", f"{len(df)} rows")
m2.metric("Avg Market Price", f"₹{int(df[target].mean()):,}")
m3.metric("Top Model", best_model)
m4.metric("Max R² Accuracy", f"{results[best_model]['R2']:.2%}")

st.divider()

# 2. ANALYSIS TABS
tab_perf, tab_viz, tab_data = st.tabs(["🎯 Model Performance", "📊 Deep Analytics", "📂 Data Explorer"])

with tab_perf:
    col_left, col_right = st.columns([1, 1])

    # Metrics Table
    res_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    col_left.subheader("Error Metrics")
    col_left.table(res_df.style.highlight_max(axis=0, subset=['R2'], color='#d4edda'))

    # R2 Comparison Bar Chart
    fig_bar = px.bar(res_df, y='R2', x=res_df.index, color='R2', title="Accuracy Score Comparison",
                     color_continuous_scale='RdYlGn')
    col_right.plotly_chart(fig_bar, use_container_width=True)

with tab_viz:
    c1, c2 = st.columns(2)

    # Feature Importance (Using Random Forest)
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance')

    fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                      title="What Drives the Price? (Feature Importance)",
                      color='Importance', color_continuous_scale='Viridis')
    c1.plotly_chart(fig_feat, use_container_width=True)

    # Actual vs Predicted Scatter
    selected_m = c2.selectbox("Select Model for Scatter Plot", list(results.keys()), index=3)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=y_actual.flatten(), y=all_preds[selected_m].flatten(),
                                     mode='markers', name='Predictions', marker=dict(opacity=0.5)))
    fig_scatter.add_trace(go.Scatter(x=[y_actual.min(), y_actual.max()], y=[y_actual.min(), y_actual.max()],
                                     line=dict(color='red', dash='dash'), name='Ideal'))
    fig_scatter.update_layout(title=f"Actual vs Predicted: {selected_m}", xaxis_title="Actual", yaxis_title="Predicted")
    c2.plotly_chart(fig_scatter, use_container_width=True)

with tab_data:
    st.dataframe(df, use_container_width=True)

# 3. PREDICTION LOGIC
if predict_btn:
    input_array = np.array([user_inputs])
    input_scaled = scaler_X.transform(input_array)

    # Use the best performing model
    final_model = trained_models[best_model]
    raw_pred = final_model.predict(input_scaled)

    # Handle shape for inverse transform
    if len(raw_pred.shape) == 1:
        raw_pred = raw_pred.reshape(-1, 1)

    final_price = scaler_y.inverse_transform(raw_pred)[0][0]

    st.balloons()
    st.success(f"### 🎯 Estimated Property Value: ₹ {int(final_price):,}")
    st.caption(f"Valuation generated using the **{best_model}** algorithm.")
