import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import time

# Backend imports
from src.data_loader import load_raw_data, save_processed_data, generate_synthetic_data
from src.preprocessing import clean_price_data, filter_target_car, encode_categorical_features, remove_outliers
from src.model import split_data, train_model, evaluate_model, save_model, get_feature_importance, calculate_advanced_metrics
from src.scraper import scrape_listings

# --- PAGE CONFIG ---
st.set_page_config(page_title="JDM Price Forecaster", page_icon="🚗", layout="wide")

# --- HEADER ---
st.title("🇯🇵 JDM Price Forecaster")
st.markdown("### Architect Your Dream Car Analysis")

# --- 1. SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio(
    "Data Source:",
    ("📁 Local CSV", "🧪 Synthetic (Demo)", "🔴 Live Data (TC-V)"),
    help="Choose where the data comes from."
)

# --- 2. DATA LOADING LOGIC ---
def get_live_data():
    progress_text = "Initializing Scraper..."
    my_bar = st.progress(0, text=progress_text)
    
    target_url = "https://www.tc-v.com/used_car/honda/fit/"
    PAGES_TO_SCRAPE = 20  # Scaled up to 20 Pages
    
    # Callback function to update progress bar from inside the scraper
    def progress_callback(current_page, total_pages):
        percent = int((current_page / total_pages) * 100)
        my_bar.progress(percent, text=f"Scraping Page {current_page}/{total_pages}...")

    # Pass the callback to the scraper
    data = scrape_listings(target_url, max_pages=PAGES_TO_SCRAPE, progress_callback=progress_callback)
    
    if not data:
        my_bar.empty()
        return None
        
    my_bar.progress(100, text="Finalizing Dataset...")
    df = pd.DataFrame(data)
    time.sleep(0.5)
    my_bar.empty()
    
    return df

def load_data(source):
    df = None
    if source == "📁 Local CSV":
        project_root = Path.cwd()
        raw_data_path = project_root / 'data' / 'raw' / 'final_cars_datasets.csv'
        if raw_data_path.exists():
            df = load_raw_data(raw_data_path)
    
    elif source == "🧪 Synthetic (Demo)":
        df = generate_synthetic_data(n_samples=1000)
        
    elif source == "🔴 Live Data (TC-V)":
        if st.sidebar.button("⚡ Scrape New Data (20 Pages)"):
            df = get_live_data()
            st.sidebar.write(f"Debug: Found {len(df) if df is not None else 0} raw rows.")
            
            if df is not None and not df.empty:
                st.session_state['live_df'] = df
            else:
                st.error("Failed to scrape data.")
        
        if 'live_df' in st.session_state:
            df = st.session_state['live_df']
        else:
            st.info("Click '⚡ Scrape New Data' to fetch live listings.")
            st.stop()

    return df

# LOAD & CLEAN
df = load_data(data_source)

if df is not None:
    df = clean_price_data(df)
else:
    st.stop()

if df.empty:
    st.error("Dataset is empty after cleaning.")
    st.stop()

# --- 3. FILTERING UI ---
available_marks = df['mark'].value_counts().index.tolist()
selected_mark = st.sidebar.selectbox("Select Make:", available_marks)

mark_mask = df['mark'] == selected_mark
available_models = df[mark_mask]['model'].value_counts().index.tolist()

selected_model = st.sidebar.selectbox("Select Model:", available_models)

# --- 4. MAIN TABS ---
tab1, tab2 = st.tabs(["📊 Data Overview", "🤖 AI Training Studio"])

# --- TAB 1: DATA OVERVIEW ---
with tab1:
    st.subheader(f"Analysis: {selected_mark.upper()} {selected_model.upper()}")
    
    df_target = filter_target_car(df, selected_mark, selected_model)
    
    if df_target.empty:
        st.warning("No cars match this filter.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Offers", len(df_target))
        
        avg_price = int(df_target['price'].mean()) if not df_target.empty else 0
        avg_mile = int(df_target['mileage'].mean()) if not df_target.empty else 0
        
        col2.metric("Avg Price (FOB)", f"{avg_price:,} '000 JPY")
        col3.metric("Avg Mileage", f"{avg_mile:,} km")
        
        st.write("Data Preview:")
        column_config = {
            "price": st.column_config.NumberColumn("Price ('000 JPY)", format="%d"),
            "mileage": st.column_config.NumberColumn("Mileage (km)", format="%d"),
            "year": st.column_config.NumberColumn("Year", format="%d")
        }
        if 'link' in df_target.columns:
            column_config["link"] = st.column_config.LinkColumn("View Car", display_text="Open Ad 🔗")
            
        st.dataframe(df_target.head(15), hide_index=True, column_config=column_config)
        
        csv_data = df_target.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Data (CSV)", csv_data, f"data_{selected_mark}.csv", "text/csv")

# --- TAB 2: AI TRAINING STUDIO ---
with tab2:
    st.subheader("🤖 AI Training Studio")
    
    if df_target.empty:
        st.warning("No data available.")
    else:
        if st.button("🚀 Start Advanced Training", type="primary"):
            
            progress_text = "Initializing Pipeline..."
            my_bar = st.progress(0, text=progress_text)
            
            # Pipeline
            my_bar.progress(10, text="Cleaning Outliers...")
            df_cleaned_ai = remove_outliers(df_target)
            
            my_bar.progress(30, text="Encoding Features...")
            df_encoded = encode_categorical_features(df_cleaned_ai)
            
            my_bar.progress(50, text="Splitting Data...")
            X_train, X_test, y_train, y_test = split_data(df_encoded)
            
            my_bar.progress(70, text="Optimizing Random Forest...")
            model = train_model(X_train, y_train)
            
            my_bar.progress(85, text="Calculating Diagnostics...")
            metrics, predictions = evaluate_model(model, X_test, y_test)
            adv_metrics = calculate_advanced_metrics(model, X_train, y_train, X_test, y_test, predictions)
            
            # SAVE STATE
            st.session_state['ai_model'] = model
            st.session_state['ai_metrics'] = metrics
            st.session_state['ai_adv_metrics'] = adv_metrics
            st.session_state['ai_predictions'] = predictions
            st.session_state['ai_y_test'] = y_test
            st.session_state['ai_X_test'] = X_test
            st.session_state['ai_cols'] = X_train.columns
            
            my_bar.progress(100, text="Done!")
            time.sleep(0.5)
            my_bar.empty()
            st.success("✅ Model Trained Successfully!")

        # DISPLAY RESULTS
        if 'ai_model' in st.session_state:
            metrics = st.session_state['ai_metrics']
            adv_metrics = st.session_state['ai_adv_metrics']
            model = st.session_state['ai_model']
            predictions = st.session_state['ai_predictions']
            y_test = st.session_state['ai_y_test']
            feature_cols = st.session_state['ai_cols']
            
            # --- 1. KEY METRICS ---
            st.markdown("#### 📊 Key Performance Indicators")
            m1, m2, m3, m4 = st.columns(4)
            
            m1.metric("Mean Error (MAE)", f"± {int(metrics['mae']):,} k JPY")
            m2.metric("Accuracy (Test R²)", f"{metrics['r2']:.2%}")
            m3.metric("RMSE", f"{int(metrics['rmse']):,} k JPY")
            
            # Overfitting Check
            train_score = adv_metrics['train_r2']
            test_score = metrics['r2']
            delta = train_score - test_score
            state_color = "normal"
            if delta > 0.15: state_color = "off" # Red flag if Train is much higher than Test
            
            m4.metric("Training Score", f"{train_score:.2%}", delta=f"{delta:.2%} Gap", delta_color="inverse")
            
            st.divider()
            
            # --- 2. ADVANCED VISUALIZATION ---
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("#### 🎯 Actual vs. Predicted")
                results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions}).sort_values(by="Actual").reset_index(drop=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(results_df.index, results_df["Actual"], color='#FF4B4B', alpha=0.5, label='Actual', s=10)
                ax.scatter(results_df.index, results_df["Predicted"], color='#1F77B4', alpha=0.5, label='Predicted', marker='x', s=10)
                ax.set_ylabel("Price ('000 JPY)")
                ax.legend()
                st.pyplot(fig)
                
            with col_viz2:
                st.markdown("#### 📉 Residuals (Error Distribution)")
                residuals = adv_metrics['residuals']
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.histplot(residuals, kde=True, color="purple", ax=ax2)
                ax2.set_xlabel("Error Magnitude (Actual - Predicted)")
                ax2.axvline(0, color='red', linestyle='--')
                st.pyplot(fig2)

            # --- 3. FEATURE IMPORTANCE ---
            st.markdown("#### 🧠 Feature Importance")
            importance_df = get_feature_importance(model, feature_cols)
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette='viridis', ax=ax3)
            st.pyplot(fig3)

            # --- 4. DOWNLOAD ---
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            st.download_button("📥 Download Trained Model (.pkl)", buffer, "jdm_model.pkl")