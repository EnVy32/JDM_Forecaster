# 🏎️ ReXie7 | JDM Price Forecaster AI

> **"Architect Your Dream Car Analysis."**

**ReXie7** is an advanced, AI-powered market analysis tool designed to forecast the prices of Japanese Domestic Market (JDM) vehicles. Built for data scientists and automotive enthusiasts, it leverages **Asynchronous Web Scraping** and **Gradient Boosting (XGBoost)** to provide real-time valuation with surgical precision.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Beta--v0.4-orange)

---

🛠️ Development Methodology: This project adopts an AI-First approach. While the architecture, logic, and problem-solving strategies are human-driven, the code implementation was accelerated using LLMs (Gemini/GPT). Every line of code has been manually reviewed, tested, and optimized to ensure reliability.

---

## 🔮 What is ReXie7?

Most price predictors use static, outdated datasets. **ReXie7** is alive. 

It connects directly to live Japanese export markets (e.g., TC-V), ingests thousands of listings in seconds using a custom AsyncIO engine, and trains a bespoke AI model on the fly. It doesn't just tell you what a car *was* worth last year; it tells you what it's worth *right now*.

### 🌟 Key Features

* **⚡ Async Neural Scraper:** Fetches 100+ pages of market data concurrently using `aiohttp`.
* **🧠 Auto-Tuned XGBoost Core:** Uses Grid Search to optimize hyperparameters specifically for the current dataset.
* **💎 Smart Grade Decoding:** Automatically categorizes messy JDM trim levels (e.g., "13G L Pkg" → "Base", "RS" → "Sport").
* **📉 Statistical Sanitization:** Uses Interquartile Range (IQR) logic to remove damaged/junk listings automatically.
* **🔮 The Oracle:** A dedicated interface to input specific specs (Year, Mileage, Grade) and receive an instant valuation.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Custom CSS / Dark Mode)
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Engineering:** Pandas, NumPy
* **Scraping:** Aiohttp, BeautifulSoup4, Asyncio

---

## 🚀 Installation & Setup (not working at the moment)

Follow these steps to deploy ReXie7 on your local machine.

### 1. Clone the Repository
~~~bash
git clone https://github.com/EnVy32/ReXie7
cd ReXie7
~~~

### 2. Create a Virtual Environment
It is recommended to keep dependencies isolated.

~~~bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
~~~

### 3. Install Dependencies
~~~bash
pip install -r requirements.txt
~~~

### 4. Ignite the System
Launch the dashboard interface.
~~~bash
streamlit run app.py
~~~

---

## 📖 User Manual

### Phase 1: Market Intel 📊
1.  Launch the app and select **"🔴 Live Market (TC-V)"** from the sidebar.
2.  Click **"⚡ Ignite Scraper"**. ReXie7 will deploy asynchronous bots to fetch the latest listings.
3.  Once complete, view the raw market data, average valuations, and inventory volume in the **Market Overview** tab.

### Phase 2: Neural Training 🧠
1.  Navigate to **"Train ReXie7"**.
2.  Click **"🚀 Train ReXie7 Model"**.
3.  Watch the pipeline visualize the process: *Cleaning -> Engineering -> Encoding -> Splitting -> Training*.
4.  Review the **KPI Dashboard** (MAE, RMSE, R² Score) to verify model accuracy.

### Phase 3: The Oracle 🔮
1.  Go to **"The Oracle"** tab.
2.  Input the specifications of a car you are looking to buy or sell (e.g., *2015 Honda Fit RS, 50,000km*).
3.  Click **"Consult ReXie7"** to get the estimated FOB price (Free On Board) in JPY.

---

## 📂 Project Structure

~~~text
ReXie7/
├── data/                   # Storage for raw and processed CSVs
├── src/
│   ├── data_loader.py      # CSV handling and synthetic data generation
│   ├── preprocessing.py    # Cleaning, IQR filtering, and One-Hot Encoding
│   ├── scraper.py          # Asyncio/Aiohttp scraping engine
│   └── model.py            # XGBoost training, evaluation, and grid search
├── app.py                  # Main Streamlit application (The UI)
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
~~~

---

## ⚠️ Disclaimer

This tool is for educational and analytical purposes only. Web scraping logic is tailored for specific market structures and may require maintenance if target websites update their DOM. Always respect `robots.txt` and rate limits (ReXie7 includes a built-in semaphore for polite crawling).

---

-Auwra 
