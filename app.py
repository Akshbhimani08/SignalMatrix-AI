import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import base64

st.set_page_config(
    page_title="SignalMatrix AI",
    layout="wide",
)

with open("logo.png", "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

# HEADER
st.markdown(f"""
<h1 style='text-align:center; display:flex; justify-content:center; align-items:center; gap:12px;'>
    <img src="data:image/png;base64,{logo_base64}" width="75" style="margin-top:-4px;">
    <span>SignalMatrix AI</span>
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:18px; color:gray;'>
AI-Powered Multi-Step Market Forecasting & Strategy Evaluation
</p>
<p style='text-align:center; font-size:13px; color:orange;'>
⚠ This model is developed strictly for educational purposes. Do NOT use it for real-world financial decisions.
</p>
<hr>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Card Container */
.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}

/* Section Titles */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Metric alignment */
.metric-card {
    text-align: center;
    padding: 15px;
    background-color: #0e1624;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


def safe_image(path, caption=None, width=900):
    if os.path.exists(path):
        st.image(path, caption=caption, width=width)
    else:
        st.warning(f"Image not found: {path}")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------


with st.expander("📖 How to Use SignalMatrix AI (Step-by-Step Guide)", expanded=False):

    st.markdown("""
### 🚀 Step 1 — Select Mode
- **Future Forecast** → Predict next 10 trading days
- **Model Evaluation** → View training performance & architecture
                
---
                
### 📂 Step-2 - Upload the data of the stocks from NIFTY50 Index(as csv formate and minimum 4 month's are needed) ,     official link to download **(https://www.nseindia.com/report-detail/eq_security)**  → Your CSV must contain:
- Date
- Open
- High
- Low
- Close  

---

### 🔮 Step 3 — Generate Forecast
- Upload file
-  Enable Validation Mode(Optional - if the data has the 80 or more no. of rows)
- Click **Run Validation** if comparing with actual data
                

---
                
### 📊 Step 4 — Interpret Results
- **Red Line** → AI Forecast
- **Orange Line** → Validation Actual (if enabled)
- **Blue Line** → Historical Data

Metrics Explained:
- **MAE** → Average prediction error
- **RMSE** → Root Mean Squared Error
- **Directional Accuracy** → % of correct trend direction

---

### 🧠 Important Notes
- Model trained on 5-Year NIFTY 50 data
- Uses 60-day lookback window
- Predicts 10-day multi-step sequence
- Educational use only
                """)


# ---------------------------------------------------
# LOAD MODEL ARTIFACTS (FAST CACHE)
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("lstm_model.keras")
    scaler = joblib.load("scaler.pkl")
    mapping = joblib.load("stock_mapping.pkl")
    return model, scaler, mapping

model, scaler, mapping = load_artifacts()

# ---------------------------------------------------
# MODE SELECTION
# ---------------------------------------------------
# ---------------------------------------------------
# MODE SELECTION (Horizontal Premium Style)
# ---------------------------------------------------

st.markdown("### Select Mode : ")

col1, col2 = st.columns(2)

with col1:
    future_btn = st.button("🚀 Future Forecast", use_container_width=True)

with col2:
    eval_btn = st.button("📊 Why Choose this model ?", use_container_width=True)

if "mode" not in st.session_state:
    st.session_state.mode = "Future Forecast"

if future_btn:
    st.session_state.mode = "Future Forecast"

if eval_btn:
    st.session_state.mode = "Model Evaluation (Training Performance)"

mode = st.session_state.mode


# ===================================================
# 🔮 FUTURE FORECAST MODE
# ===================================================

if mode == "Future Forecast":

    st.markdown(
    '<div class="section-header">@10-Day Multi-Step Forecast</div>',
    unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
    "Upload Stock File (.csv or .xlsx)"
    )

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Please upload CSV or Excel file.")
            st.stop()

        st.markdown('</div>', unsafe_allow_html=True)


        df.columns = df.columns.str.strip()

        COLUMN_MAPPING = {
            "Open Price": "Open",
            "High Price": "High",
            "Low Price": "Low",
            "Close Price": "Close",
        }

        df = df.rename(columns=COLUMN_MAPPING)

        required_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            st.error("CSV must contain Open, High, Low, Close columns.")
            st.stop()

        for col in required_cols:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""),
                errors="coerce"
            )

        if "Date" not in df.columns:
            st.error("CSV must contain a 'Date' column.")
            st.stop()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.sort_values("Date")

        # -----------------------------
        # INDICATORS
        # -----------------------------
        df["MA_20"] = df["Close"].rolling(20).mean()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        df["BB_Middle"] = df["Close"].rolling(20).mean()
        std = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Middle"] + 2 * std
        df["BB_Lower"] = df["BB_Middle"] - 2 * std

        df = df.dropna()

        if len(df) < 80:
            st.error("Minimum 80 rows required.")
            st.stop()

        # -----------------------------
        # FEATURES
        # -----------------------------


        # ---------------------------------
        # Detect Stock Symbol
        # ---------------------------------

        if "Symbol" in df.columns:
            stock_symbol = str(df["Symbol"].iloc[0])
        else:
            # Try extracting from filename
            filename = uploaded_file.name
            parts = filename.split("-")
            stock_symbol = parts[-3] if len(parts) >= 3 else "Unknown"

        df["Symbol"] = mapping.get(stock_symbol, 0)


        features = [
            'Symbol',
            'Open','High','Low','Close',
            'MA_20','EMA_20','RSI',
            'MACD','Signal_Line',
            'BB_Middle','BB_Upper','BB_Lower'
        ]

        close_index = features.index("Close")

        # -----------------------------
        # VALIDATION CONTROLS
        # -----------------------------
        validation_mode = st.checkbox("Enable Validation Mode (Compare with Actual)")

        if "run_validation" not in st.session_state:
            st.session_state.run_validation = False

        if validation_mode:
            if st.button("Run Validation"):
                st.session_state.run_validation = True

        n_future = 10

        # =====================================================
        # VALIDATION PIPELINE
        # =====================================================
        if validation_mode and st.session_state.run_validation:

            if len(df) < 60 + n_future:
                st.error("Not enough data for validation.")
                st.stop()

            split_index = len(df) - n_future

            input_window = df[features].iloc[split_index-60:split_index].values
            actual_future = df["Close"].iloc[split_index:].values
            future_dates = df["Date"].iloc[split_index:].values

        else:
            input_window = df[features].tail(60).values
            actual_future = None
            last_date = df["Date"].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=n_future
            )

        # -----------------------------
        # PREDICTION
        # -----------------------------
        input_scaled = scaler.transform(input_window)
        X_input = input_scaled.reshape(1, 60, len(features))

        pred_scaled = model.predict(X_input, verbose=0)[0]

        future_prices = []
        for val in pred_scaled:
            dummy = np.zeros(len(features))
            dummy[close_index] = val
            inv = scaler.inverse_transform([dummy])[0]
            future_prices.append(inv[close_index])


        last_close = df["Close"].iloc[-1]
        pred_last = future_prices[0]

        # 1️⃣ Additive correction
        bias = df["Close"].iloc[-1] - future_prices[0]
        future_prices = [p + bias for p in future_prices]

        # 2️⃣ Light clamp to prevent explosion
        future_series = pd.Series(future_prices).ewm(span=3).mean()
        future_prices = future_series.values

        # -----------------------------
        # METRICS DISPLAY
        # -----------------------------
        # -----------------------------
        # FORECAST SUMMARY CARD
        # -----------------------------
        with st.container():
            st.markdown(
                """
                <div style="
                    background-color:#111827;
                    padding:25px;
                    border-radius:15px;
                    box-shadow:0px 4px 20px rgba(0,0,0,0.3);
                    margin-bottom:25px;
                ">
                <div style="font-size:22px;font-weight:600;margin-bottom:20px;">
                📊 Forecast Summary
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Last Close", f"₹ {df['Close'].iloc[-1]:,.2f}")

            with col2:
                st.metric("Final Predicted Price", f"₹ {future_prices[-1]:,.2f}")

            st.markdown("</div>", unsafe_allow_html=True)



        # -----------------------------
        # PLOT
        # -----------------------------

        
        # -----------------------------
        # CREATE PLOT
        # -----------------------------
        fig = go.Figure()

        st.markdown(f'<div class="section-title">📌 Stock: {stock_symbol}</div>', unsafe_allow_html=True)

        fig.add_trace(go.Scatter(
            x=df["Date"].tail(60),
            y=df["Close"].tail(60),
            name="Historical",
            line=dict(color="blue", width=3)
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            name="Forecast (10 Days)",
            line=dict(color="red", width=3)
        ))

        if validation_mode and st.session_state.run_validation:

            from sklearn.metrics import mean_absolute_error, mean_squared_error

            mae = mean_absolute_error(actual_future, future_prices)
            rmse = np.sqrt(mean_squared_error(actual_future, future_prices))

            actual_direction = np.sign(np.diff(actual_future))
            pred_direction = np.sign(np.diff(future_prices))
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100


            # -----------------------------
            # VALIDATION METRICS CARD
            # -----------------------------
            with st.container():
                st.markdown(
                    """
                    <div style="
                        background-color:#111827;
                        padding:25px;
                        border-radius:15px;
                        box-shadow:0px 4px 20px rgba(0,0,0,0.3);
                        margin-bottom:25px;
                    ">
                    <div style="font-size:22px;font-weight:600;margin-bottom:20px;">
                    📈 Validation Metrics
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("MAE", f"{mae:.2f}")

                with c2:
                    st.metric("RMSE", f"{rmse:.2f}")


                with c3:
                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            padding:15px;
                            background-color:#0e1624;
                            border-radius:12px;
                        ">
                            <div style="font-size:14px; color:gray;">
                                Directional Accuracy
                            </div>
                            <div style="
                                font-size:33px;
                                font-weight:700;
                                color:#228B22;
                            ">
                                {directional_accuracy:.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                st.markdown("</div>", unsafe_allow_html=True)


            fig.add_trace(go.Scatter(
                x=future_dates,
                y=actual_future,
                name="Validation Actual",
                line=dict(color="orange", width=3)
            ))

            fig.update_layout(
                template="plotly_dark",
                height=600,
                yaxis_title="Prices",
                xaxis_title="Date"
            )


        st.plotly_chart(fig, use_container_width=True)

        


# =============================================================
# MODEL EVALUATION SECTION
# =============================================================
if mode == "Model Evaluation (Training Performance)":

    st.markdown("## 📘 Model Evaluation While Training & Dataset Overview")

    # AI credibility statement
    st.info(""" **Detailed overview about the Model and DataSet:**  
    We used the LSTM(Long Short Term Memory-a type of RNN) a Seq. 2 Seq. deep learning model is trained on 5 years of historical NIFTY 50 Index data approx 1.75 Lakh rows(combined stocks data of NIFTY50 Index) training data and 45k(combined stocks data of NIFTY50 Index) rows for validation
    using daily OHLC price structure and advanced technical indicators.The model learns temporal price behavior using a 60-day lookback window and 
    predicts the next 10-day movement pattern and also validates if the actual data for the prediction is provided.
    """)

    st.markdown("---")

    # ----------------------------
    # MODEL ARCHITECTURE
    # ----------------------------
    st.markdown("### 🧠 Model Architecture")

    safe_image("model_architecture.png", "LSTM Model Architecture Summary")

    st.markdown("""
    - 2 Stacked LSTM Layers (128 + 64 units)
    - Dropout Regularization
    - Dense Layers for Pattern Extraction
    - 10-Step Multi-Day Forecast Output
    - Total Parameters: **124,522**
    """)

    st.markdown("---")

    # ----------------------------
    # TRAINING PERFORMANCE METRICS
    # ----------------------------
    st.markdown('<div class="section-title">📊 Training Performance Metrics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("RMSE", "135.78")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("MAE", "58.60")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("MAPE", "14.23%")
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("R² Score", "0.9766")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



    # ----------------------------
    # TRAINING VISUALIZATION
    # ----------------------------
    st.markdown("### 📈 Training Visualization")

    safe_image("training_actual_vs_pred.png", 
"Actual vs Predicted — Full Training Dataset")

    safe_image("last_100_days.png", 
"Last 100 Days Prediction Behavior")

    st.markdown("---")

    # ----------------------------
    # FEATURE DESCRIPTION SECTION
    # ----------------------------
    st.markdown("## 📚 Feature Engineering & Input Variables")

    st.markdown("""
The model uses 13 engineered features derived from price action and technical indicators.
Below is the description of each feature:
""")

    feature_descriptions = {
        "Symbol": "Encoded identifier representing the NIFTY 50 Index.",
        "Open": "Opening price of the trading day.",
        "High": "Highest price reached during the day.",
        "Low": "Lowest price during the trading session.",
        "Close": "Final closing price of the day.",
        "MA_20": "20-Day Simple Moving Average — captures medium-term trend.",
        "EMA_20": "20-Day Exponential Moving Average — gives more weight to recent prices.",
        "RSI": "Relative Strength Index — measures overbought/oversold conditions.",
        "MACD": "Moving Average Convergence Divergence — identifies momentum shifts.",
        "Signal_Line": "Smoothed MACD signal for crossover detection.",
        "BB_Middle": "Middle Bollinger Band — 20-day average.",
        "BB_Upper": "Upper Bollinger Band — volatility resistance level.",
        "BB_Lower": "Lower Bollinger Band — volatility support level."
    }

    for feature, desc in feature_descriptions.items():
        st.markdown(f"**{feature}** → {desc}")

    st.markdown("---")

# ---------------------------------------------------
# STICKY FOOTER
# ---------------------------------------------------

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    font-size: 13px;
    color: gray;
    background-color: #0e1117;
}
</style>

<div class="footer">
Made with ❤️ by Aksh Bhimani
</div>
""", unsafe_allow_html=True)

