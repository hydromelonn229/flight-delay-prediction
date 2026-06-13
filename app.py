import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(
    page_title="Flight Delay Predictor & Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style Injection for Premium Aesthetics
st.markdown("""
<style>
    /* Main Layout Styling */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Remove default Streamlit top padding and space */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Hide default Streamlit white header space but keep the toggle button visible */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    [data-testid="stHeader"] button {
        color: #f8fafc !important;
    }
    [data-testid="stHeader"] svg {
        fill: #f8fafc !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar text readability */
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #94a3b8 !important;
        font-weight: 600;
        margin-bottom: 10px !important;
    }
    [data-testid="stSidebar"] li {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Polishing the Sidebar Navigation (Changing Radio list to Menu Button Tabs) */
    div[data-testid="stRadio"] label {
        display: block !important;
        width: 100% !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    /* Hide the radio selection circles */
    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    
    /* Remove spacing on inner container */
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] {
        padding-left: 0px !important;
    }
    
    /* Hover state */
    div[data-testid="stRadio"] label:hover {
        background-color: rgba(56, 189, 248, 0.08) !important;
        color: #38bdf8 !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
    }
    
    /* Selected/Active state styling via :has pseudo-class */
    div[data-testid="stRadio"] label:has(input:checked) {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
        font-weight: 600 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.1) !important;
    }
    
    /* Hide default radio header label */
    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
        display: none !important;
    }
    
    /* Title and Header Typography */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Transparent metric card container */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: all 0.3s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(30, 144, 255, 0.4);
        box-shadow: 0 12px 40px 0 rgba(30, 144, 255, 0.15);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Prediction Banners */
    .pred-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.4) 100%);
        border: 1px solid #ef4444;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    }
    .pred-banner-low {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.4) 100%);
        border: 1px solid #22c55e;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* Interactive elements styling */
    .stSelectbox, .stSlider, .stNumberInput {
        color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load and cache raw EDA data
@st.cache_data
def load_eda_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "data", "Airline Dataset.csv")
    df = pd.read_csv(csv_path)
    df["Departure Date"] = pd.to_datetime(df["Departure Date"])
    df["Month"] = df["Departure Date"].dt.month
    return df

# Helper: Load and cache raw ML data
@st.cache_data
def load_ml_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "data", "Airline_Delay_Cause.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["arr_flights", "arr_del15"])
    df = df[df["arr_flights"] >= 20]
    return df

# Helper: Load ML models
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    lr_model = joblib.load(os.path.join(base_dir, "models", "model_lr.joblib"))
    rf_model = joblib.load(os.path.join(base_dir, "models", "model_rf.joblib"))
    return lr_model, rf_model

# Load resources
df_eda = load_eda_data()
df_ml = load_ml_data()
model_lr, model_rf = load_models()

# Map month integers to labels globally
month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# Sidebar Setup
st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>Flight Delay Predictor and Analytics</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Data Mining Project Dashboard</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation Menu
page = st.sidebar.radio(
    "Go To Page",
    ["Analytics Dashboard", "Delay Predictor", "Model Insights & Trade-offs"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Executive Summary")
st.sidebar.markdown("""
This application visualizes airport delay metrics and predicts high-delay months (>= 20%) using trained machine learning classification models.
- EDA Dataset: 98,619 Flights
- ML Dataset: 155,952 Monthly Records
""")

# ==========================================
# PAGE 1: ANALYTICS DASHBOARD
# ==========================================
if page == "Analytics Dashboard":
    st.markdown("<h1>Flight Delay Analytics (EDA)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Exploratory analysis of flight-level statistics and operational profiles</p>", unsafe_allow_html=True)
    
    # Precompute metrics
    total_flights = len(df_eda)
    unique_airports = df_eda["Airport Name"].nunique()
    unique_countries = df_eda["Country Name"].nunique()
    
    tot_del = df_ml['arr_del15'].sum()
    tot_delay_min = df_ml['arr_delay'].sum()
    avg_delay_duration = tot_delay_min / tot_del if tot_del > 0 else 0.0

    # Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Flights Analyzed</div>
            <div class="metric-value">{total_flights:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Airports Represented</div>
            <div class="metric-value">{unique_airports:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Countries Connected</div>
            <div class="metric-value">{unique_countries:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg. Historical Delay</div>
            <div class="metric-value">{avg_delay_duration:.1f}m</div>
        </div>
        """, unsafe_allow_html=True)

    # Status & Monthly Trends Grid
    g1_col1, g1_col2 = st.columns(2)
    
    with g1_col1:
        st.subheader("Flight Status Distribution")
        status_counts = df_eda["Flight Status"].value_counts().to_dict()
        dist = {
            "Cancelled": status_counts.get("Cancelled", 0),
            "On Time": status_counts.get("On Time", 0),
            "Delayed": status_counts.get("Delayed", 0)
        }
        fig_dist = px.pie(
            names=list(dist.keys()),
            values=list(dist.values()),
            hole=0.45,
            color=list(dist.keys()),
            color_discrete_map={
                "On Time": "#22c55e",
                "Delayed": "#f97316",
                "Cancelled": "#ef4444"
            }
        )
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with g1_col2:
        st.subheader("Seasonal Delay Trends")
        
        monthly_stats = df_eda.groupby("Month")["Flight Status"].apply(
            lambda x: pd.Series({
                "total": len(x),
                "delayed": (x == "Delayed").sum()
            })
        ).unstack().reset_index()
        monthly_stats["rate"] = (monthly_stats["delayed"] / monthly_stats["total"]) * 100
        monthly_stats["Month Name"] = monthly_stats["Month"].map(month_names)
        
        fig_trend = px.line(
            monthly_stats,
            x="Month Name",
            y="rate",
            markers=True,
            labels={"rate": "Delay Rate (%)", "Month Name": "Month"},
            color_discrete_sequence=["#38bdf8"]
        )
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', ticksuffix="%")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    
    # Airport Analysis Section (Resolving the Sample Size Paradox via Bayesian Smoothing)
    st.subheader("Airport Performance: Delay Risk Index (DRI)")
    st.markdown("""
    > [!IMPORTANT]
    > **Resolving the Sample Size Paradox:** 
    > - Ranking airports purely by raw **delay rate (%)** highlights tiny airports with low flight volumes (e.g. 100% delay rate on only 3 flights).
    > - Ranking by **delay count** highlights major traffic hubs simply because they handle massive flight volumes.
    > 
    > To balance both, we use a **Delay Risk Index (DRI)** (calculated via Bayesian smoothing). This formula pulls small-sample airports toward the global average, highlighting hubs that have **both high flight volumes and high delay rates** (representing true operational bottlenecks).
    """)
    
    airport_counts = df_eda.groupby("Airport Name")["Flight Status"].apply(
        lambda x: pd.Series({
            "total": len(x),
            "delayed": (x == "Delayed").sum()
        })
    ).unstack()
    overall_delay_rate_val = (df_eda["Flight Status"] == "Delayed").mean()
    m_param = 11
    airport_counts["weighted_score"] = (airport_counts["delayed"] + m_param * overall_delay_rate_val) / (airport_counts["total"] + m_param)
    
    top_airports_weighted = airport_counts.sort_values(by="weighted_score", ascending=False).head(10).reset_index()
    top_airports_weighted["rate"] = (top_airports_weighted["delayed"] / top_airports_weighted["total"]) * 100
    top_airports_weighted["weighted_score"] = top_airports_weighted["weighted_score"] * 100

    st.markdown("##### Top 10 Airports by Delay Risk Index")
    
    fig_air = px.bar(
        top_airports_weighted,
        x="weighted_score",
        y="Airport Name",
        orientation="h",
        text=top_airports_weighted["weighted_score"].apply(lambda x: f"{x:.1f}%"),
        hover_data={"total": True, "delayed": True, "rate": ":.1f%"},
        labels={
            "weighted_score": "Delay Risk Index (DRI)",
            "Airport Name": "Airport Name",
            "total": "Total Flights",
            "delayed": "Delayed Flights",
            "rate": "Raw Delay Rate (%)"
        },
        color="weighted_score",
        color_continuous_scale="Oranges"
    )
    fig_air.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_air, use_container_width=True)

    st.markdown("---")
    
    # Continent Analysis Section
    st.subheader("Delay Rates by Airport Continent")
    
    continent_counts = df_eda.groupby("Continents")["Flight Status"].apply(
        lambda x: pd.Series({
            "total": len(x),
            "delayed": (x == "Delayed").sum()
        })
    ).unstack().reset_index()
    continent_counts["rate"] = (continent_counts["delayed"] / continent_counts["total"]) * 100

    fig_cont = px.bar(
        continent_counts,
        x="Continents",
        y="rate",
        text=continent_counts["rate"].apply(lambda x: f"{x:.1f}%"),
        labels={"rate": "Delay Rate (%)", "Continents": "Continent"},
        color="rate",
        color_continuous_scale="Purples"
    )
    fig_cont.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        coloraxis_showscale=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', ticksuffix="%")
    )
    st.plotly_chart(fig_cont, use_container_width=True)

# ==========================================
# PAGE 2: DELAY PREDICTOR
# ==========================================
elif page == "Delay Predictor":
    st.markdown("<h1>Monthly Delay Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Predict the likelihood of a carrier-airport pair experiencing a high delay month (>= 20%)</p>", unsafe_allow_html=True)
    
    # Form layout
    st.markdown("### Input Flight Parameters")
    
    col_inp1, col_inp2 = st.columns(2)
    
    with col_inp1:
        # Carrier Select box (searchable)
        carrier_names_dict = df_ml.set_index("carrier")["carrier_name"].to_dict()
        carriers_list = [{"code": code, "name": name} for code, name in sorted(carrier_names_dict.items())]
        carrier_dict = {c["name"]: c["code"] for c in carriers_list}

        selected_carrier_name = st.selectbox(
            "Select Airline (Carrier)",
            list(carrier_dict.keys()),
            help="Select the airline company code."
        )
        carrier_code = carrier_dict[selected_carrier_name]
        
        # Airport Select box (searchable)
        airport_names_dict = df_ml.set_index("airport")["airport_name"].to_dict()
        airports_list = [{"code": code, "name": name} for code, name in sorted(airport_names_dict.items())]
        airport_dict = {f"{a['code']} - {a['name']}": a['code'] for a in airports_list}

        selected_airport_label = st.selectbox(
            "Select Destination Airport",
            list(airport_dict.keys()),
            help="Select the destination airport code."
        )
        airport_code = airport_dict[selected_airport_label]
        
        # Month Slider
        month = st.slider("Select Month of Operation", min_value=1, max_value=12, value=6)
        
    with col_inp2:
        # Arriving Flights Number
        arr_flights = st.number_input(
            "Average Monthly Flights scheduled on Route",
            min_value=1,
            max_value=25000,
            value=100,
            step=10,
            help="Higher flight volume generally increases congestion risk."
        )
        
        # Cancellation Rate Slider
        cancel_rate_pct = st.slider(
            "Expected Monthly Cancellation Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
            help="Percentage of scheduled flights that get cancelled. This is a strong indicator of operational distress."
        )
        
        # Diversion Rate Slider
        divert_rate_pct = st.slider(
            "Expected Monthly Diversion Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.5,
            step=0.1,
            help="Percentage of flights diverted to other airports."
        )
        
    st.markdown("---")
    st.markdown("### Model Selection & Configuration")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        model_type = st.selectbox(
            "Select Machine Learning Model",
            ["Logistic Regression", "Random Forest"],
            help="Logistic Regression yields high recall (catches more delays). Random Forest yields higher precision (fewer false warnings)."
        )
        
    with col_m2:
        if model_type == "Random Forest":
            rf_threshold = st.slider(
                "Classification Probability Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.35,
                step=0.05,
                help="Adjusting this threshold allows tuning the tradeoff between recall and precision. Notebook adjusted baseline is 0.35."
            )
        else:
            st.info("Logistic Regression is configured with class-weights balanced (runs at default probability threshold 0.50).")
            rf_threshold = 0.50

    # Execute Prediction
    # Create feature vectors
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    log_arr_flights = np.log1p(arr_flights)
    cancel_rate = cancel_rate_pct / 100.0
    divert_rate = divert_rate_pct / 100.0
    
    query_df = pd.DataFrame({
        "month_sin": [month_sin],
        "month_cos": [month_cos],
        "log_arr_flights": [log_arr_flights],
        "cancel_rate": [cancel_rate],
        "divert_rate": [divert_rate],
        "carrier": [carrier_code],
        "airport": [airport_code]
    })
    
    # Calculate probabilities
    if model_type == "Logistic Regression":
        pred_prob = model_lr.predict_proba(query_df)[0][1]
        is_high_delay = 1 if pred_prob >= 0.50 else 0
    else:
        pred_prob = model_rf.predict_proba(query_df)[0][1]
        is_high_delay = 1 if pred_prob >= rf_threshold else 0
        
    # Result Showcase
    st.markdown("### Prediction Results")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if is_high_delay == 1:
            st.markdown(f"""
            <div class="pred-banner-high">
                <h3 style="color: #ef4444; margin: 0;">HIGH DELAY EXPECTED</h3>
                <p style="margin: 10px 0 0 0; color: #fecaca; font-size: 0.95rem;">
                    Delay probability exceeds decision threshold. Significant congestion or scheduling risk.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-banner-low">
                <h3 style="color: #22c55e; margin: 0;">NORMAL DELAY RATE</h3>
                <p style="margin: 10px 0 0 0; color: #dcfce7; font-size: 0.95rem;">
                    Operational conditions look stable. Under 20% delay rate expected.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Gauge representation
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Delay Risk Probability", 'font': {'color': '#f8fafc', 'size': 16}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc", 'ticksuffix': "%"},
                'bar': {'color': "#38bdf8" if is_high_delay == 0 else "#ef4444"},
                'bgcolor': "rgba(30, 41, 59, 0.5)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(34, 197, 94, 0.1)'},
                    {'range': [20, 50], 'color': 'rgba(234, 179, 8, 0.1)'},
                    {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': (0.50 if model_type == "Logistic Regression" else rf_threshold) * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=260,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with res_col2:
        st.markdown("#### Input Profile Breakdown")
        
        # Build comparative dataframe
        profile_data = {
            "Parameter": [
                "Airline (Carrier)",
                "Destination Airport",
                "Operating Month",
                "Flight Volume count",
                "Log Transformed Volume",
                "Scheduled Cancellations",
                "Scheduled Diversions"
            ],
            "Value": [
                f"{selected_carrier_name} ({carrier_code})",
                f"{selected_airport_label}",
                f"{month_names[month]}",
                f"{arr_flights} flights",
                f"{log_arr_flights:.4f}",
                f"{cancel_rate_pct:.1f}% ({round(arr_flights * cancel_rate)} flights)",
                f"{divert_rate_pct:.1f}% ({round(arr_flights * divert_rate)} flights)"
            ],
            "Risk Assessment": [
                "Carrier-specific base effect",
                "Airport congestion category",
                "Cyclical seasonality impact",
                "Traffic load volume",
                "Volume scale factor",
                "Strong indicator of bottlenecking" if cancel_rate_pct > 3.0 else "Normal cancellation range",
                "Minor operational stress indicator" if divert_rate_pct > 1.0 else "Standard diversion range"
            ]
        }
        st.table(pd.DataFrame(profile_data))
        
        st.markdown("#### Operational Insights for this Route")
        if cancel_rate_pct > 4.0:
            st.warning("High Cancellation Warning: Cancellation rates above 4% are heavily associated with systemic carrier delays. Expect domino effects on downstream schedules.")
        if arr_flights > 800:
            st.info("Airport Volume Congestion: High schedule volumes (>800 monthly flights) place this carrier-airport pair in a higher baseline delay profile due to tarmac/runway traffic constraints.")
        if month in [6, 7, 8, 12]:
            st.info("High Seasonality Period: Operating during summer travel rushes (Jun-Aug) or winter storms (Dec) increases delay probabilities across all carriers.")

# ==========================================
# PAGE 3: MODEL INSIGHTS & PERFORMANCE
# ==========================================
else:
    st.markdown("<h1>Model Insights & Trade-offs</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8;'>Deep-dive details on training, testing, and operational deployment metrics</p>", unsafe_allow_html=True)
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.subheader("Feature Importances (Random Forest)")
        st.markdown("This chart displays the strength of each parameter in predicting high delay months.")
        
        # Precomputed importances
        imp_data = pd.DataFrame({
            "feature": [
                "Cancellation Rate",
                "Airport Volume (Log Flights)",
                "Winter Seasonality (Month Cos)",
                "Summer Seasonality (Month Sin)",
                "Diversion Rate",
                "Delta Air Lines (DL)",
                "Frontier Airlines (F9)",
                "JetBlue Airways (B6)"
            ],
            "importance": [0.1451, 0.1351, 0.0703, 0.0632, 0.0465, 0.0280, 0.0138, 0.0120]
        })
        
        fig_imp = px.bar(
            imp_data.sort_values(by="importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Relative Importance", "feature": "Model Feature"},
            color="importance",
            color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("""
        **Operational Takeaway:** 
        Notice that **Cancellation Rate** and **Airport Traffic Volume** dwarf carrier identities (e.g. DL, F9, B6). 
        This confirms that operational congestion and airport volume are far more critical to schedule maintenance than the airline itself.
        """)
        
    with col_ins2:
        st.subheader("Precision vs. Recall Trade-off")
        st.markdown("""
        When deploying monthly delay predictions, operations managers must balance sensitivity against alarm costs:
        
        - **Logistic Regression (Sensitivity Focus):**
          - **Recall: 74%** | **Precision: 55%** | **F1: 63%**
          - *Best when missing a delay is costly.* Used for scheduling standby crews or staging support infrastructure. Captures almost 3/4ths of delayed months, but has a high false alarm rate.
          
        - **Random Forest (Specificity Focus):**
          - **Recall: 42%** | **Precision: 70%** | **F1: 52%**
          - *Best when false alarms are costly.* Used for passenger warnings or flight cancellation plans. When it warns of a delay, it is correct 70% of the time, but misses over half of the actual delay months.
        """)
        
        # Heatmap layout of confusion matrices
        cm_type = st.radio("Display Model Confusion Matrix (Test Set)", ["Logistic Regression", "Random Forest"])
        
        if cm_type == "Logistic Regression":
            cm_data = [[4657, 2527], [1073, 3031]]
        else:
            cm_data = [[6431, 753], [2384, 1720]]
            
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Class", y="True Class", color="Count"),
            x=['Normal Rate (0)', 'High Delay (1)'],
            y=['Normal Rate (0)', 'High Delay (1)'],
            text_auto=True,
            color_continuous_scale="Blues" if cm_type == "Logistic Regression" else "Greens"
        )
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            coloraxis_showscale=False,
            height=300
        )
        st.plotly_chart(fig_cm, use_container_width=True)
