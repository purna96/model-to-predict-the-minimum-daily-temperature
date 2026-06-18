"""
app.py — Streamlit App for Time Series Temperature Forecasting
Capstone Project 2: Minimum Daily Temperature Prediction — Melbourne, Australia
"""

import warnings
warnings.filterwarnings("ignore")

import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🌡️ Melbourne Temp Forecast",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Dark glassmorphism theme */
  .stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #ecf0f1;
    font-family: 'Inter', sans-serif;
  }
  [data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85);
    backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,0.08);
  }
  .metric-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease;
  }
  .metric-card:hover { transform: translateY(-3px); }
  .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #2ecc71, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .metric-label {
    font-size: 0.85rem;
    color: #bdc3c7;
    margin-top: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ecf0f1;
    border-left: 4px solid #2ecc71;
    padding-left: 12px;
    margin: 24px 0 12px;
  }
  .badge-good  { background:#2ecc71; color:#000; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
  .badge-warn  { background:#f39c12; color:#000; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
  .badge-bad   { background:#e74c3c; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
  div[data-testid="column"] { gap: 12px; }
  .stSelectbox label, .stSlider label { color: #bdc3c7 !important; font-weight: 500; }
  h1 { background: linear-gradient(90deg, #2ecc71, #3498db, #9b59b6);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
       font-size: 2.5rem !important; font-weight: 800 !important; }
  h2, h3 { color: #ecf0f1 !important; }
  .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("🌡️ Melbourne Temperature Forecaster")
st.markdown(
    "<p style='color:#bdc3c7;font-size:1.05rem;margin-top:-12px;'>"
    "Predict minimum daily temperature using 18 time series models | "
    "<b style='color:#2ecc71;'>Capstone Project 2</b></p>",
    unsafe_allow_html=True
)
st.divider()


# ─── Utility ─────────────────────────────────────────────────────────────────

DATA_URL  = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
DATA_FILE = "daily-minimum-temperatures.csv"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.04)",
    font=dict(color="#ecf0f1", family="Inter"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showgrid=True),
)

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_FILE):
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")
    df.columns = ["Temp"]
    df = df.sort_index()
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range).ffill()
    df["Temp"] = df["Temp"].interpolate(method="time")
    Q1, Q3 = df["Temp"].quantile(0.25), df["Temp"].quantile(0.75)
    IQR = Q3 - Q1
    df["Temp"] = df["Temp"].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)
    return df

@st.cache_data(show_spinner=False)
def load_comparison():
    paths = ["models/comparison_ranked.csv", "models/all_results.csv"]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    path = f"models/{name.replace(' ','_')}_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

def create_features(df_in):
    d = df_in.copy()
    d["day_of_year"] = d.index.dayofyear
    d["day_of_week"] = d.index.dayofweek
    d["month"]       = d.index.month
    d["quarter"]     = d.index.quarter
    d["year"]        = d.index.year
    for k in [1, 2, 3]:
        d[f"sin_{k}"] = np.sin(2 * np.pi * k * d["day_of_year"] / 365.25)
        d[f"cos_{k}"] = np.cos(2 * np.pi * k * d["day_of_year"] / 365.25)
    for lag in [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 365]:
        d[f"lag_{lag}"] = d["Temp"].shift(lag)
    for w in [7, 14, 30, 60, 90]:
        shifted = d["Temp"].shift(1)
        d[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        d[f"roll_std_{w}"]  = shifted.rolling(w).std()
        d[f"roll_min_{w}"]  = shifted.rolling(w).min()
        d[f"roll_max_{w}"]  = shifted.rolling(w).max()
    d["ewm_7"]  = d["Temp"].shift(1).ewm(span=7).mean()
    d["ewm_30"] = d["Temp"].shift(1).ewm(span=30).mean()
    return d.dropna()


# ─── Load Data ───────────────────────────────────────────────────────────────

with st.spinner("Loading dataset..."):
    df = load_data()

split_idx = int(len(df) * 0.80)
train_series = df["Temp"].iloc[:split_idx]
test_series  = df["Temp"].iloc[split_idx:]

comp_df = load_comparison()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")

    st.markdown("**📊 Model Selection**")
    all_models = [
        "AR", "MA", "ARMA", "ARIMA", "SARIMA", "Holt-Winters",
        "Linear Regression", "Random Forest", "XGBoost", "LightGBM", "SVR", "Prophet",
        "RNN", "LSTM", "GRU", "Transformer", "TFT", "N-BEATS"
    ]
    selected_models = st.multiselect(
        "Choose models to display:",
        options=all_models,
        default=["SARIMA", "LightGBM", "LSTM", "Prophet"]
    )

    st.markdown("---")
    st.markdown("**📅 Dataset Info**")
    st.info(f"""
- **Records:** {len(df):,} days
- **Train:** {len(train_series):,} days
- **Test:**  {len(test_series):,} days
- **Split:** {test_series.index[0].date()}
- **Range:** 1981 – 1990
""")

    st.markdown("---")
    st.markdown("**🔮 Forecast Horizon**")
    horizon_days = st.slider("Days to forecast ahead:", 7, 365, 60, step=7)

    st.markdown("---")
    st.markdown("**📈 Rolling Window**")
    roll_window = st.slider("Rolling average window (days):", 7, 90, 30)

    st.markdown("---")
    st.caption("Capstone Project 2 | Time Series Forecasting")
    st.caption("Dataset: Melbourne Daily Min Temp (1981–1990)")


# ─── Tab Layout ──────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Data Explorer",
    "🏆 Model Comparison",
    "🔮 Forecasting",
    "📊 EDA Dashboard",
    "ℹ️ About"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">📈 Historical Temperature Data</div>', unsafe_allow_html=True)

    # KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = {
        "Mean Temp": f"{df['Temp'].mean():.2f}°C",
        "Min Temp":  f"{df['Temp'].min():.2f}°C",
        "Max Temp":  f"{df['Temp'].max():.2f}°C",
        "Std Dev":   f"{df['Temp'].std():.2f}°C",
        "Records":   f"{len(df):,}",
    }
    for col, (label, val) in zip([c1, c2, c3, c4, c5], stats.items()):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Full time series with rolling average
    roll_avg = df["Temp"].rolling(roll_window).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Temp"],
        name="Daily Temp", line=dict(color="rgba(52,152,219,0.4)", width=0.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Temp: %{y:.1f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=roll_avg.index, y=roll_avg,
        name=f"{roll_window}-Day Rolling Avg",
        line=dict(color="#2ecc71", width=2.2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Rolling Avg: %{y:.1f}°C<extra></extra>"
    ))
    fig.add_vrect(
        x0=test_series.index[0], x1=test_series.index[-1],
        fillcolor="#e74c3c", opacity=0.06,
        annotation_text="Test Set (20%)", annotation_position="top left",
        annotation_font=dict(color="#e74c3c", size=11)
    )
    fig.update_layout(
        title="Melbourne Daily Minimum Temperature (1981–1990)",
        yaxis_title="Temperature (°C)", xaxis_title="Date",
        height=400, **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly heatmap
    st.markdown('<div class="section-header">📅 Monthly Temperature Heatmap</div>', unsafe_allow_html=True)
    df_pivot = df.copy()
    df_pivot["Year"]  = df_pivot.index.year
    df_pivot["Month"] = df_pivot.index.month
    monthly_avg = df_pivot.groupby(["Year", "Month"])["Temp"].mean().unstack()
    monthly_avg.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_heat = px.imshow(
        monthly_avg, color_continuous_scale="RdBu_r",
        labels=dict(color="Temp (°C)"),
        title="Monthly Average Temperature Heatmap"
    )
    fig_heat.update_layout(height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">🏆 Model Comparison Dashboard</div>', unsafe_allow_html=True)

    if comp_df is not None and not comp_df.empty:
        # Ensure required columns
        display_cols = [c for c in ["Model","Test MAE","Test RMSE","Test MAPE","Test R2","Train R2","Overfit?"] if c in comp_df.columns]
        comp_show = comp_df[display_cols].copy() if display_cols else comp_df

        # Style table
        def style_overfit(val):
            if str(val) == "Yes": return "background-color: rgba(231,76,60,0.3); color: #e74c3c;"
            return "background-color: rgba(46,204,113,0.15); color: #2ecc71;"

        st.dataframe(
            comp_show.style.applymap(style_overfit, subset=["Overfit?"])
                           .format({c: "{:.4f}" for c in comp_show.select_dtypes("float").columns}),
            use_container_width=True, height=520
        )

        # Charts row
        if "Test MAE" in comp_df.columns and "Model" in comp_df.columns:
            c1, c2 = st.columns(2)

            with c1:
                fig_mae = px.bar(
                    comp_df.sort_values("Test MAE"),
                    x="Test MAE", y="Model", orientation="h",
                    color="Test MAE", color_continuous_scale="Teal",
                    title="Test MAE by Model (lower = better)"
                )
                fig_mae.update_layout(height=500, **PLOTLY_LAYOUT, coloraxis_showscale=False)
                st.plotly_chart(fig_mae, use_container_width=True)

            with c2:
                if "Train R2" in comp_df.columns and "Test R2" in comp_df.columns:
                    fig_r2 = go.Figure()
                    fig_r2.add_trace(go.Bar(
                        x=comp_df["Model"], y=comp_df["Train R2"],
                        name="Train R²", marker_color="#3498db", opacity=0.8
                    ))
                    fig_r2.add_trace(go.Bar(
                        x=comp_df["Model"], y=comp_df["Test R2"],
                        name="Test R²", marker_color="#e74c3c", opacity=0.8
                    ))
                    fig_r2.update_layout(
                        title="Train vs Test R² (Overfitting Check)",
                        barmode="group", xaxis_tickangle=-45, height=500,
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.info("🔄 Run the notebook first to generate `models/comparison_ranked.csv`. The comparison table will appear here automatically.")
        st.markdown("""
        ### Model Categories

        | Category | Models |
        |---|---|
        | **Traditional Statistical** | AR, MA, ARMA, ARIMA, SARIMA, Holt-Winters |
        | **Machine Learning** | Linear Regression, Random Forest, XGBoost, LightGBM, SVR, Prophet |
        | **Deep Learning** | RNN, LSTM, GRU, Transformer, TFT, N-BEATS |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Forecasting
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">🔮 Interactive Forecast Explorer</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("#### 🎛️ Forecast Settings")
        forecast_model = st.selectbox(
            "Select Model:", all_models, index=all_models.index("SARIMA")
        )
        show_ci = st.toggle("Show 95% Confidence Interval", value=True)
        show_train = st.toggle("Show Training History", value=True)
        zoom_test = st.toggle("Zoom to Test Period", value=False)

    # Simple statistical forecast using Holt-Winters or SARIMA-like approach
    # (using rolling mean as a demonstration — actual models loaded from pkl if available)
    with col_left:
        # Naive seasonal forecast for display purposes
        # We'll use a simple but realistic seasonal decomposition approach
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        @st.cache_resource(show_spinner=False)
        def build_hw_forecast(_train_series):
            try:
                hw = ExponentialSmoothing(
                    _train_series, trend="add", seasonal="add", seasonal_periods=365
                ).fit(optimized=True, use_brute=False)
                return hw
            except Exception:
                return None

        hw_fc_model = build_hw_forecast(train_series)

        if hw_fc_model is not None:
            fitted_tr = hw_fc_model.fittedvalues
            pred_te   = hw_fc_model.forecast(steps=len(test_series))
            future    = hw_fc_model.forecast(steps=len(test_series) + horizon_days)[-horizon_days:]
            future_dates = pd.date_range(
                start=test_series.index[-1] + timedelta(days=1), periods=horizon_days
            )

            residuals = test_series.values - pred_te.values
            std_resid = np.std(residuals)

            mae, rmse, mape, r2 = compute_metrics(test_series.values, pred_te.values)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            for col, label, val, fmt in [
                (m1, "MAE",     f"{mae:.3f}°C",  "#2ecc71"),
                (m2, "RMSE",    f"{rmse:.3f}°C", "#3498db"),
                (m3, "MAPE",    f"{mape:.2f}%",  "#f39c12"),
                (m4, "R²",      f"{r2:.4f}",     "#9b59b6"),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="background:linear-gradient(90deg,{fmt},{fmt}aa);-webkit-background-clip:text;">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("")

            # Forecast plot
            fig_fc = go.Figure()

            if show_train:
                x_range = test_series.index if zoom_test else train_series.index
                y_range = fitted_tr[test_series.index[0]:] if zoom_test else fitted_tr
                if not zoom_test:
                    fig_fc.add_trace(go.Scatter(
                        x=train_series.index, y=train_series.values,
                        name="Train (Actual)", line=dict(color="rgba(52,152,219,0.5)", width=0.8)
                    ))

            fig_fc.add_trace(go.Scatter(
                x=test_series.index, y=test_series.values,
                name="Test (Actual)", line=dict(color="#2ecc71", width=2)
            ))
            fig_fc.add_trace(go.Scatter(
                x=test_series.index, y=pred_te.values,
                name="Test (Predicted)", line=dict(color="#e74c3c", width=2, dash="dash")
            ))

            if show_ci:
                upper = pred_te.values + 1.96 * std_resid
                lower = pred_te.values - 1.96 * std_resid
                fig_fc.add_trace(go.Scatter(
                    x=list(test_series.index) + list(test_series.index[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill="toself", fillcolor="rgba(231,76,60,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI"
                ))

            # Future forecast
            fig_fc.add_trace(go.Scatter(
                x=future_dates, y=future.values,
                name=f"Future ({horizon_days}d)",
                line=dict(color="#f39c12", width=2.5, dash="dot"),
            ))
            fig_fc.add_vrect(
                x0=future_dates[0], x1=future_dates[-1],
                fillcolor="rgba(243,156,18,0.06)"
            )

            fig_fc.update_layout(
                title=f"Holt-Winters Forecast | Test Period + {horizon_days}-Day Future",
                yaxis_title="Temperature (°C)", xaxis_title="Date",
                height=450, **PLOTLY_LAYOUT
            )
            if zoom_test:
                fig_fc.update_xaxes(range=[test_series.index[0], future_dates[-1]])
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.warning("Could not build forecast model. Ensure statsmodels is installed.")

    # Future forecast table
    if hw_fc_model is not None:
        st.markdown('<div class="section-header">📋 Future Forecast Table</div>', unsafe_allow_html=True)
        future_df = pd.DataFrame({
            "Date":            pd.DatetimeIndex(future_dates),
            "Forecast (°C)":   future.values.round(2),
            "Lower 95% CI":    (future.values - 1.96 * std_resid).round(2),
            "Upper 95% CI":    (future.values + 1.96 * std_resid).round(2),
        })
        st.dataframe(future_df, use_container_width=True, height=300)
        csv = future_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV", csv, "temperature_forecast.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df["Temp"], nbinsx=45, name="Daily Temp",
            marker_color="#3498db", opacity=0.8,
            hovertemplate="Temp: %{x:.1f}°C<br>Count: %{y}<extra></extra>"
        ))
        fig_dist.add_vline(x=df["Temp"].mean(), line_color="#e74c3c",
                           line_dash="dash", line_width=2,
                           annotation_text=f"Mean={df['Temp'].mean():.1f}°C")
        fig_dist.add_vline(x=df["Temp"].median(), line_color="#2ecc71",
                           line_dash="dot", line_width=2,
                           annotation_text=f"Median={df['Temp'].median():.1f}°C")
        fig_dist.update_layout(title="Temperature Distribution", height=340, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        # Monthly boxplot
        df_box = df.copy()
        df_box["Month"] = df_box.index.month
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        df_box["Month_Name"] = df_box["Month"].map(month_names)

        fig_box = px.box(
            df_box, x="Month_Name", y="Temp",
            category_orders={"Month_Name": list(month_names.values())},
            color="Month_Name", title="Temperature by Month",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_box.update_layout(showlegend=False, height=340, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)

    # Year-over-year
    df_yoy = df.copy()
    df_yoy["Year"]  = df_yoy.index.year
    df_yoy["Month"] = df_yoy.index.month
    monthly_avg_yoy = df_yoy.groupby(["Year","Month"])["Temp"].mean().reset_index()
    monthly_avg_yoy["Month_Name"] = monthly_avg_yoy["Month"].map(month_names)

    fig_yoy = px.line(
        monthly_avg_yoy, x="Month_Name", y="Temp", color="Year",
        category_orders={"Month_Name": list(month_names.values())},
        title="Year-over-Year Monthly Average",
        markers=True, color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_yoy.update_layout(height=380, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_yoy, use_container_width=True)

    # Seasonal decomposition info
    c3, c4 = st.columns(2)
    with c3:
        # ACF-like display using rolling correlation
        lags = list(range(1, 61))
        acf_vals = [df["Temp"].autocorr(lag=l) for l in lags]

        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF",
                                  marker_color="#3498db", opacity=0.8))
        fig_acf.add_hline(y=0, line_color="white", line_width=0.8)
        conf = 1.96 / np.sqrt(len(df))
        fig_acf.add_hline(y=conf, line_color="#e74c3c", line_dash="dash",
                          annotation_text="95% CI")
        fig_acf.add_hline(y=-conf, line_color="#e74c3c", line_dash="dash")
        fig_acf.update_layout(title="Autocorrelation Function (ACF)", height=330, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_acf, use_container_width=True)

    with c4:
        # Rolling statistics
        r7  = df["Temp"].rolling(7).mean()
        r30 = df["Temp"].rolling(30).mean()
        r90 = df["Temp"].rolling(90).mean()
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=df.index, y=df["Temp"], name="Daily",
                                      line=dict(color="rgba(189,195,199,0.35)", width=0.7)))
        fig_roll.add_trace(go.Scatter(x=r7.index, y=r7, name="7-Day MA",
                                      line=dict(color="#e74c3c", width=1.5)))
        fig_roll.add_trace(go.Scatter(x=r30.index, y=r30, name="30-Day MA",
                                      line=dict(color="#2ecc71", width=2)))
        fig_roll.add_trace(go.Scatter(x=r90.index, y=r90, name="90-Day MA",
                                      line=dict(color="#f39c12", width=2.5)))
        fig_roll.update_layout(title="Rolling Moving Averages", height=330, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_roll, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — About
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## 📘 About This Project")
    st.markdown("""
    ### 🌡️ Time Series Forecasting — Melbourne Minimum Daily Temperature

    **Objective:** Build and compare 18 time series models to predict daily minimum temperature in Melbourne, Australia.

    ---

    ### 📦 Dataset
    - **Source:** [Kaggle — Daily Minimum Temperatures in Melbourne](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne)
    - **Period:** January 1, 1981 – December 31, 1990 (3,650 days)
    - **Feature:** Single univariate series — `minimum temperature (°C)`

    ---

    ### 🔀 Data Split
    | Split | Records | Period |
    |---|---|---|
    | **Train (80%)** | ~2,920 days | 1981 – 1988 |
    | **Test (20%)**  | ~730 days   | 1988 – 1990 |

    > ⚠️ Chronological split is used (no shuffling) to preserve temporal integrity.

    ---

    ### 🧩 Models Implemented
    | Category | Models |
    |---|---|
    | **Traditional Statistical** | AR, MA, ARMA, ARIMA, SARIMA, Holt-Winters |
    | **Machine Learning** | Linear Regression, Random Forest, XGBoost, LightGBM, SVR, Prophet |
    | **Deep Learning** | RNN, LSTM, GRU, Transformer, TFT, N-BEATS |

    ---

    ### 📏 Evaluation Metrics
    | Metric | Description |
    |---|---|
    | **MAE** | Mean Absolute Error — average prediction error in °C |
    | **RMSE** | Root Mean Squared Error — penalizes large errors more |
    | **MAPE** | Mean Absolute Percentage Error — relative error % |
    | **R²** | Coefficient of Determination — variance explained (1.0 = perfect) |

    ---

    ### ⚙️ Tech Stack
    `Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `XGBoost` · `LightGBM` · `Prophet` · `TensorFlow/Keras` · `Streamlit` · `Plotly`
    """)
