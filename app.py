import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import requests
import os
from streamlit_lottie import st_lottie

# Backend imports
from src.data_loader import load_raw_data
from src.preprocessing import clean_price_data, simplify_grades, encode_categorical_features, remove_outliers
from src.model import split_data, train_model, evaluate_model, get_feature_importance, calculate_advanced_metrics
from src.scraper import scrape_listings

# --- [CONFIG & ASSETS] ---
st.set_page_config(
    page_title="ReXie7 | JDM Systems", 
    page_icon="🔰", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache Directory Setup
CACHE_DIR = Path.cwd() / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE = CACHE_DIR / "session_cache.csv"

# Load Lottie Animation
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_car = load_lottieurl("https://lottie.host/5a706691-1402-4660-a8f8-27e1c2780e8e/0wQ6YyF7Cj.json")
lottie_brain = load_lottieurl("https://lottie.host/6d2d3855-6d00-4742-8700-6cb4f227183e/q5F2d7iW72.json") 

# --- [CYBERPUNK STYLING] ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
    }
    
    /* Top Bar Styling */
    header {visibility: hidden;}
    
    /* Containers & Cards */
    div[data-testid="stExpander"] {
        border: 1px solid #333;
        border-radius: 8px;
        background-color: #0e1117;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Courier New', monospace;
        color: #00FFFF !important; /* Cyan Neon */
        text-shadow: 0 0 10px #00FFFF;
    }
    div[data-testid="stMetricLabel"] {
        color: #888;
        font-weight: bold;
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #111;
        color: #00FF99;
        border: 1px solid #333;
    }
    .stSelectbox > div > div > div {
        background-color: #111;
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #0e1117, #1f2937);
        color: #00FFFF;
        border: 1px solid #00FFFF;
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #00FFFF;
        color: black;
        box-shadow: 0 0 15px #00FFFF;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: #888;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e1117;
        color: #FF00FF !important; /* Pink Neon */
        border-bottom: 2px solid #FF00FF;
    }
    
    /* Custom Status Box */
    .stStatus {
        border: 1px solid #FF00FF;
        background-color: #110011;
    }
    
    /* Context Bar (Current Car) */
    .context-box {
        padding: 10px;
        border-bottom: 1px solid #333;
        margin-bottom: 20px;
        background-color: #0e1117;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    </style>
    """, unsafe_allow_html=True)

# --- [HELPER: EXTRACT CAR NAME] ---
def get_car_name_from_df(df):
    """
    Extracts the most frequent Mark and Model from the dataframe.
    """
    if df is None or df.empty:
        return "UNKNOWN ASSET"
    try:
        # Get the most common mark and model (Mode)
        top_mark = df['mark'].mode()[0]
        top_model = df['model'].mode()[0]
        return f"{top_mark} {top_model}".upper()
    except:
        return "UNKNOWN ASSET"

# --- [SESSION PERSISTENCE LOGIC] ---
if 'buffer_df' not in st.session_state:
    if SESSION_FILE.exists():
        try:
            st.session_state.buffer_df = pd.read_csv(SESSION_FILE)
            st.session_state.buffer_logs = ["✅ Session Restored from Cache."]
            # Auto-detect name from loaded file
            st.session_state.target_name = get_car_name_from_df(st.session_state.buffer_df)
        except Exception:
            st.session_state.buffer_df = None
            st.session_state.buffer_logs = []
            st.session_state.target_name = None
    else:
        st.session_state.buffer_df = None
        st.session_state.buffer_logs = []
        st.session_state.target_name = None

if 'buffer_model' not in st.session_state: st.session_state.buffer_model = None

# --- [HEADER] ---
c1, c2 = st.columns([1, 10])
with c1:
    st.markdown("## 🔰")
with c2:
    st.markdown("<h1 style='margin-bottom:0; color: white;'>ReXie7 <span style='color:#FF00FF; font-size:0.5em;'>// JDM ANALYTICS CORE</span></h1>", unsafe_allow_html=True)

st.markdown("---")

# --- [MISSION CONTROL] ---
with st.expander("🎛️ MISSION CONTROL & DATA FEED", expanded=True):
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    
    with col_input1:
        presets = {
            "Mazda RX-7 (FD3S)": "https://www.tc-v.com/used_car/mazda/rx-7/",
            "Honda S2000": "https://www.tc-v.com/used_car/honda/s2000/",
            "Nissan Skyline GT-R": "https://www.tc-v.com/used_car/nissan/skyline%20gt-r/",
            "Toyota Supra": "https://www.tc-v.com/used_car/toyota/supra/",
            "Subaru Impreza WRX STI": "https://www.tc-v.com/used_car/subaru/impreza%20wrx%20sti/",
            "Mitsubishi Lancer Evo": "https://www.tc-v.com/used_car/mitsubishi/lancer%20evolution/",
            "Custom URL 🔗": "CUSTOM"
        }
        
        target_choice = st.selectbox("Select Asset Class", list(presets.keys()))
        
        if target_choice == "Custom URL 🔗":
            target_url = st.text_input("Enter TC-V URL", "https://www.tc-v.com/used_car/...")
        else:
            target_url = presets[target_choice]
            st.caption(f"Targeting: {target_url}")

    with col_input2:
        scan_depth = st.slider("Scan Depth (Pages)", min_value=1, max_value=100, value=10)
    
    with col_input3:
        st.write("") 
        st.write("") 
        ignite_btn = st.button("⚡ INITIALIZE SCAN")

# --- [LOGIC CORE] ---
uplink_container = st.empty()

if ignite_btn:
    with uplink_container.container():
        lottie_col1, lottie_col2 = st.columns([1, 2])
        with lottie_col1:
             if lottie_car: st_lottie(lottie_car, height=150, key="loading_car")
        with lottie_col2:
            st.markdown("### 📡 ESTABLISHING UPLINK...")
            st.markdown(f"Targeting: `{target_choice}`")
            status_text = st.empty()
            
            def update_progress(current, total):
                status_text.code(f"scannning_node_buffer: [{current}/{total}] packets received")

            # SCRAPE
            data, logs = scrape_listings(target_url, max_pages=scan_depth, progress_callback=update_progress)
            
            if data:
                df = pd.DataFrame(data)
                st.session_state.buffer_df = df
                st.session_state.buffer_logs = logs
                
                # EXTRACT REAL NAME FROM DATA
                real_name = get_car_name_from_df(df)
                st.session_state.target_name = real_name
                
                df.to_csv(SESSION_FILE, index=False)
                status_text.success(f"DATA SECURE. IDENTIFIED: {real_name}")
                time.sleep(1) 
            else:
                st.error("CONNECTION FAILED. NO ASSETS FOUND.")
                with st.expander("View System Logs"):
                    st.write(logs)
                st.stop()

if st.session_state.buffer_df is not None:
    uplink_container.empty()

if st.session_state.buffer_df is None:
    st.info("⚠️ Awaiting Mission Parameters. Select a car and click Initialize Scan.")
    st.stop()

df = st.session_state.buffer_df
df = clean_price_data(df)

# --- [CONTEXT BAR (Global HUD)] ---
st.markdown(f"""
<div style="background-color: #111; padding: 15px; border-radius: 5px; border-left: 5px solid #00FFFF; margin-bottom: 20px;">
    <span style="color: #888; font-weight: bold; margin-right: 10px;">ACTIVE ASSET:</span>
    <span style="color: white; font-weight: bold; font-size: 1.2em; margin-right: 30px;">{st.session_state.target_name}</span>
    <span style="color: #888; font-weight: bold; margin-right: 10px;">DATABASE VOLUME:</span>
    <span style="color: #00FFFF; font-weight: bold; font-size: 1.2em;">{len(df)} UNITS</span>
</div>
""", unsafe_allow_html=True)


# --- [DASHBOARD UI] ---
tab_market, tab_ai, tab_oracle = st.tabs(["📊 MARKET TELEMETRY", "🧠 NEURAL TRAINING", "🔮 ORACLE PREDICTION"])

# === TAB 1: MARKET TELEMETRY ===
with tab_market:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("ASSETS TRACKED", len(df))
    
    # FORMATTING FIX: Multiply by 1000 to show full JPY
    avg_price = int(df['price'].mean() * 1000)
    kpi2.metric("AVG VALUATION", f"{avg_price:,} JPY")
    
    kpi3.metric("AVG MILEAGE", f"{int(df['mileage'].mean()):,} km")
    
    unicorns = len(df[(df['price'] < df['price'].quantile(0.25)) & (df['mileage'] < df['mileage'].quantile(0.25))])
    kpi4.metric("POTENTIAL DEALS", unicorns, delta="High ROI")

    st.markdown("### 📈 DEPRECIATION VECTORS")
    
    # Note: We display raw price ('000 JPY) in charts to keep axis clean, 
    # but the metrics above now show full price.
    try:
        fig = px.scatter(
            df, x="mileage", y="price", color="year",
            hover_data=['grade', 'transmission'],
            trendline="ols", trendline_color_override="#FF00FF",
            title="VALUATION MATRIX: PRICE ('000 JPY) vs MILEAGE",
            color_continuous_scale="Viridis", template="plotly_dark"
        )
    except:
        fig = px.scatter(
            df, x="mileage", y="price", color="year",
            title="VALUATION MATRIX: PRICE ('000 JPY) vs MILEAGE (Insufficient data for Trendline)",
            color_continuous_scale="Viridis", template="plotly_dark"
        )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white'), opacity=0.8))
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font=dict(family="Courier New, monospace", color="#e0e0e0"))
    st.plotly_chart(fig, use_container_width=True)

    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        fig2 = px.histogram(df, x="price", nbins=20, title="PRICE DISTRIBUTION", template="plotly_dark", color_discrete_sequence=['#00FFFF'])
        fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        st.plotly_chart(fig2, use_container_width=True)
    
    with c_chart2:
        trans_counts = df['transmission'].value_counts()
        fig3 = px.pie(names=trans_counts.index, values=trans_counts.values, title="TRANSMISSION RATIO", template="plotly_dark", hole=0.4, color_discrete_sequence=['#FF00FF', '#00FFFF'])
        fig3.update_layout(paper_bgcolor="#0e1117")
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📂 RAW DATA_LOGS", expanded=False):
        st.dataframe(df, use_container_width=True)

# === TAB 2: NEURAL TRAINING ===
with tab_ai:
    st.markdown("### 🧬 MODEL CALIBRATION")
    
    col_train_btn, col_train_status = st.columns([1, 3])
    with col_train_btn:
        if st.button("🚀 INITIATE TRAINING SEQUENCE"):
            
            with st.status("🧠 Neural Pathways Initializing...", expanded=True) as status:
                
                st.write("🔹 Sanitizing Inputs (IQR Logic)...")
                time.sleep(0.8)
                df_clean = remove_outliers(df)
                
                st.write("🔹 Engineering Grade Features & Encoding...")
                time.sleep(0.8)
                df_eng = simplify_grades(df_clean)
                df_enc = encode_categorical_features(df_eng)
                
                cols_to_drop = ['link', 'mark', 'model', 'Unnamed: 0']
                df_enc = df_enc.drop(columns=[c for c in cols_to_drop if c in df_enc.columns], errors='ignore')
                
                st.write("🔹 Splitting Vector Space (80/20)...")
                time.sleep(0.6)
                X_train, X_test, y_train, y_test = split_data(df_enc)
                
                st.write("⚡ Optimizing XGBoost Hyperparameters (Grid Search)...")
                model = train_model(X_train, y_train)
                
                st.write("🔹 Running Diagnostics & Calculating Overfit Risk...")
                metrics, preds = evaluate_model(model, X_test, y_test)
                adv_metrics = calculate_advanced_metrics(model, X_train, y_train, X_test, y_test, preds)
                
                st.session_state.buffer_model = {
                    'model': model,
                    'metrics': metrics,
                    'adv_metrics': adv_metrics,
                    'cols': X_train.columns
                }
                status.update(label="✅ Training Complete. Neural Core Online.", state="complete", expanded=False)

    if st.session_state.buffer_model:
        data_model = st.session_state.buffer_model
        met = data_model['metrics']
        adv = data_model['adv_metrics']
        
        overfit_gap = (adv['train_r2'] - met['r2']) * 100
        overfit_color = "normal" if overfit_gap < 10 else "inverse"
        
        imp_df = get_feature_importance(data_model['model'], data_model['cols'])
        if not imp_df.empty:
            top_feature = imp_df.iloc[0]['Feature']
            top_importance = imp_df.iloc[0]['Importance']
        else:
            top_feature = "N/A"
            top_importance = 0
        
        # KPI ROW
        m1, m2, m3, m4 = st.columns(4)
        
        # FORMATTING FIX: Multiply by 1000
        mae_full = int(met['mae'] * 1000)
        m1.metric("PRECISION (MAE)", f"± {mae_full:,} JPY")
        
        m2.metric("ACCURACY (R²)", f"{met['r2']:.1%}")
        
        m3.metric(
            "OVERFIT RISK", 
            f"{overfit_gap:.1f}%", 
            delta=f"{'⚠️ High Risk' if overfit_gap > 10 else '✅ Stable'}",
            delta_color=overfit_color
        )
        
        m4.metric("TOP DRIVER", top_feature.upper().replace("_", " "), f"{top_importance:.1%} Impact")

        st.markdown("---")
        
        c_plots1, c_plots2 = st.columns([2, 1])
        with c_plots1:
            fig_imp = px.bar(
                imp_df.head(10), x="Importance", y="Feature", orientation='h', 
                title="NEURAL WEIGHTS (Which specs drive price?)", template="plotly_dark", 
                color="Importance", color_continuous_scale="Bluered"
            )
            fig_imp.update_layout(
                yaxis={'categoryorder':'total ascending'}, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(family="Courier New, monospace", color="#e0e0e0")
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with c_plots2:
            residuals = adv['residuals']
            fig_res = px.histogram(
                residuals, nbins=30, title="ERROR DISTRIBUTION (Residuals)",
                template="plotly_dark", color_discrete_sequence=['#FF00FF']
            )
            fig_res.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", showlegend=False)
            st.plotly_chart(fig_res, use_container_width=True)

# === TAB 3: THE ORACLE ===
with tab_oracle:
    st.markdown("### 🔮 VALUATION ENGINE")
    
    if not st.session_state.buffer_model:
        st.warning("⚠️ NEURAL NETWORK OFFLINE. Please Train Model in previous tab.")
    else:
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            in_year = st.slider("Model Year", 1990, 2025, 2000)
            in_mileage = st.slider("Mileage (km)", 0, 200000, 80000)
            in_engine = st.selectbox("Engine Disp. (cc)", [660, 1300, 2000, 2500, 3000, 3800])
        with col_o2:
            in_trans = st.radio("Transmission", ["MT (Manual)", "AT (Auto)"])
            in_grade = st.selectbox("Trim Level", ["Sport (Type R/RZ/STI)", "Base", "Luxury", "Standard/Other"])
            in_drive = st.radio("Drivetrain", ["2WD", "4WD"])

        if st.button("🔮 GENERATE PREDICTION"):
            input_dict = {
                'year': [in_year],
                'mileage': [in_mileage],
                'engine_capacity': [in_engine],
                'transmission': ['mt' if "MT" in in_trans else 'at'],
                'drive': ['4wd' if "4WD" in in_drive else '2wd'],
                'grade_category': [in_grade],
                'fuel': ['gasoline'],
                'hand_drive': ['rhd']
            }
            
            input_df = pd.DataFrame(input_dict)
            input_enc = pd.get_dummies(input_df)
            
            model_cols = st.session_state.buffer_model['cols']
            final_input = input_enc.reindex(columns=model_cols, fill_value=0)
            
            # Predict (Result is in '000 JPY)
            pred_thousands = st.session_state.buffer_model['model'].predict(final_input)[0]
            
            # Convert to Full JPY
            pred_full = int(pred_thousands * 1000)
            
            st.markdown("---")
            c_res1, c_res2 = st.columns([1, 3])
            with c_res1:
                st.metric("ESTIMATED VALUE", f"{pred_full:,} JPY")
            with c_res2:
                st.info("Valuation represents FOB (Free On Board) price at Japanese port.")