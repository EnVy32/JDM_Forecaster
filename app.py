import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import modules
from src.data_loader import load_raw_data, save_processed_data, generate_synthetic_data
from src.preprocessing import clean_price_data, filter_target_car, encode_categorical_features, remove_outliers 
from src.model import split_data, train_model, evaluate_model

#Website configuration
st.set_page_config(page_title="JDM Price Forecaster", page_icon="🚗", layout= "wide")

#Header
st.title("🇯🇵 JDM Price Forecaster")
st.markdown("### Architect Your Dream Car Analysis")

#Loading data
#Using cache for performance 
@st.cache_data
def get_data():
    project_root = Path.cwd()
    raw_data_path = project_root / 'data' / 'raw' / 'cars_datasets.csv'

    if not raw_data_path.exists():
        return None

    df = load_raw_data(raw_data_path)
    df = clean_price_data(df)
    return df

df = get_data()

if df is None:
    st.error("Error: No data in folder data/raw/!")
    st.stop()

#User Interface (Sidebar)
st.sidebar.header("⚙️ Configuration")

#Data Source Toggle
use_synthetic = st.sidebar.toggle("🧪 Use Synthetic (Demo) Data", value=False, help="Switch to generated data to test the AI pipeline logic.")

#Data loading
def get_data(use_synthetic_mode):
    if use_synthetic_mode:
        #Load fake perfect data
        return generate_synthetic_data(n_samples=1000)
    else:
        #Load the real CSV
        project_root = Path.cwd()
        raw_data_path = project_root / 'data' / 'raw' / 'cars_datasets.csv'
        if not raw_data_path.exists():
            return None
        return load_raw_data(raw_data_path)
    
#Load data based on toggle
df = get_data(use_synthetic)
#Clean it immediatly 
df= clean_price_data(df)

if df is None:
    st.error("ERROR: Raw data file not found!")
    st.stop()


#A. Mark selection
available_marks = df["mark"].value_counts().index.tolist()
selected_mark = st.sidebar.selectbox("Select brand:", available_marks)

#B. Model selection
mark_mask = df["mark"] == selected_mark
available_models = df[mark_mask]["model"].value_counts().index.tolist()

selected_model = st.sidebar.selectbox("Select model:", available_models)

#Main content
tab1, tab2 = st.tabs(["📊 Data Overview", "🤖 AI Training Studio"])

#Tab1
with tab1:
    st.subheader(f"Analysis: {selected_mark.upper()} {selected_model.upper()}")

    #Filter
    df_target = filter_target_car(df, selected_mark, selected_model)

    #Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Offer count", len(df_target))
    col2.metric("Average price", f"{int(df_target['price'].mean() * 1000):,} JPY")
    col3.metric("Average mileage", f"{int(df_target['mileage'].mean())}km")

    st.dataframe(df_target.head(), hide_index= True)

    #Download button
    csv_data = df_target.to_csv(index = False).encode('utf-8')

    st.download_button(
        label = "📥 Download Cleaned Data (CSV)",
        data = csv_data,
        file_name = f"cleaned_{selected_mark}_{selected_model}.csv",
        mime ="text/csv", 
        help = "Download this dataset to open it in Excel."
    )

    st.divider()
    st.subheader("🔍 Data Diagnostics")
    
    # Calculate Correlation
    # We only select numeric columns
    numeric_df = df_target.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    
    # Display Price Correlation
    st.write("How much do features influence Price? (-1 to 1)")
    st.dataframe(corr[['price']].sort_values(by='price', ascending=False).style.background_gradient(cmap='coolwarm'))
    
    st.info("💡 Tip: If 'year' or 'mileage' is close to 0, the AI cannot learn anything.")

# --- TAB 2: AI TRAINING STUDIO ---
# --- TAB 2: AI TRAINING STUDIO ---
with tab2:
    st.subheader("🤖 AI Training Studio")
    st.markdown("Train a Random Forest model and analyze its decision-making process.")
    
    # Display warning if dataset is small
    if len(df_target) < 50:
        st.warning("⚠️ Warning: Very small dataset. AI performance might be unstable.")
    
    if st.button("🚀 Start Advanced Training", type="primary"):
        
        # --- 1. PRE-PROCESSING ---
        progress_text = "Initializing Training Pipeline..."
        my_bar = st.progress(0, text=progress_text)
        
        # A. Outlier Removal
        my_bar.progress(10, text="Cleaning Data (Removing Outliers)...")
        df_cleaned_ai = remove_outliers(df_target)
        removed_rows = len(df_target) - len(df_cleaned_ai)
        if removed_rows > 0:
            st.info(f"🧹 Removed {removed_rows} outliers (Price < 200k or Engine size mismatch).")
        
        # B. Encoding
        my_bar.progress(30, text="Encoding Categorical Features...")
        df_encoded = encode_categorical_features(df_cleaned_ai)
        
        # C. Splitting
        my_bar.progress(50, text="Splitting Data (Train/Test)...")
        X_train, X_test, y_train, y_test = split_data(df_encoded)
        
        # --- 2. TRAINING ---
        my_bar.progress(70, text="Training Random Forest Model...")
        # Note: train_model now returns a RandomForestRegressor
        model = train_model(X_train, y_train)
        
        # --- 3. EVALUATION ---
        my_bar.progress(90, text="Calculating Advanced Metrics...")
        mae, r2, predictions = evaluate_model(model, X_test, y_test)
        
        my_bar.progress(100, text="Training Complete!")
        my_bar.empty() # Remove progress bar
        
        st.success("✅ Model Trained Successfully!")
        
        # --- 4. ADVANCED DASHBOARD ---
        
        # SECTION A: Key Metrics
        st.markdown("### 📊 Model Scorecard")
        kpi1, kpi2, kpi3 = st.columns(3)
        
        mae_real = int(mae * 1000)
        kpi1.metric("Mean Absolute Error (MAE)", f"± {mae_real:,} JPY", help="Average error in prediction.")
        kpi2.metric("Accuracy (R² Score)", f"{r2:.2%}", help="How well the model explains the variance.")
        kpi3.metric("Test Samples", len(y_test), help="Number of cars used for this test.")

        st.divider()

        # SECTION B: Feature Importance (The "Brain" Scan)
        st.markdown("### 🧠 Feature Importance")
        st.write("Which factors influenced the price the most?")
        
        # Extract importance from Random Forest
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X_train.columns
            
            # Create DataFrame for plotting
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10) # Top 10
            
            # Plot using Seaborn
            fig_feat, ax_feat = plt.subplots(figsize=(10, 4))
            sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis', ax=ax_feat)
            ax_feat.set_title("Top 10 Drivers of Price")
            st.pyplot(fig_feat)
        else:
            st.warning("Feature importance not available for this model type.")

        st.divider()

        # SECTION C: Error Diagnostics (The "Health" Check)
        col_diag1, col_diag2 = st.columns(2)
        
        with col_diag1:
            st.markdown("#### 📉 Actual vs. Predicted")
            # Visualizing alignment
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.scatter(y_test, predictions, alpha=0.5, color='blue')
            # Perfect line
            min_val = min(y_test.min(), predictions.min())
            max_val = max(y_test.max(), predictions.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax1.set_xlabel("Actual Price")
            ax1.set_ylabel("Predicted Price")
            ax1.set_title("Linearity Check")
            st.pyplot(fig1)

        with col_diag2:
            st.markdown("#### 🔔 Residuals Distribution")
            # Visualizing the error spread
            residuals = y_test - predictions
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(residuals, kde=True, color='purple', ax=ax2)
            ax2.axvline(x=0, color='red', linestyle='--')
            ax2.set_title("Error Distribution (Should be centered at 0)")
            st.pyplot(fig2)
            
        st.info("💡 Analysis Tip: If 'Feature Importance' shows 'Random' columns (like un-encoded index) as high, check data cleaning. If 'Residuals' are skewed left/right, the model is biased.")