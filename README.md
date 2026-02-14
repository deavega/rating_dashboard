# 🌍 Sovereign Credit Watch (Fitch SRM Replicator)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dashboardsrm.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Active-success)

**Sovereign Credit Watch** is an interactive analytical dashboard designed to replicate the **Fitch Ratings Sovereign Rating Model (SRM)**. It allows analysts, economists, and researchers to reverse-engineer sovereign credit scores, visualize peer comparisons, and simulate the impact of economic policy changes on a country's credit rating.

**Hosting server:** [live dashboard](https://dashboardsrm.streamlit.app/)

## 🚀 Motivation

Replicating a Sovereign Rating Model is traditionally difficult because:
1.  **Data Fragmentation:** The 18+ input variables (WGI, IMF WEO, World Bank, etc.) are scattered across different databases with varying update cycles.
2.  **Calculation Complexity:** Re-calculating centered averages, log transformations, and weights manually is error-prone.

**The Solution:**
This dashboard streamlines the process by ingesting a **single data source**—the *Fitch Sovereign Data Comparator*—to instantly generate Shadow Ratings, visualize Qualitative Overlays (QO), and perform stress testing.

---

## ✨ Key Features

* **📊 SRM Score Calculator:** Precise replication of the 0–16.00 scoring system using the 18-variable regression model.
* **🌍 Peer Comparison:** Benchmarking tool to compare a specific country against peers with similar ratings (e.g., Indonesia vs. India/Philippines).
* **🧪 Policy Simulation:** "What-if" analysis tool. Modify indicators (e.g., increase Debt/GDP or decrease FX Reserves) to see real-time impacts on the credit rating.
* **📄 PDF Sentiment Analysis:** NLP-based module that scans Fitch Rating Action Commentaries (PDFs) to detect specific "Upgrade" or "Downgrade" sensitivity triggers.
* **📉 Qualitative Overlay (QO) Detector:** Automatically calculates the gap between the Model output and the Actual Rating to identify structural adjustments.

---

## 📂 Data Requirements

> ⚠️ **Important Note on Data Sources**
>
> This dashboard **does not** fetch live data from public APIs (World Bank/IMF) due to data consistency issues and API limits.
>
> **Required Input:**
> To use this dashboard, you must possess the **Fitch Sovereign Data Comparator** file (format: `.xlsb` or `.xlsx`). This file is typically available to Fitch Connect subscribers.
>
> **Why this approach?**
> * Ensures 100% alignment with the official rating committee data.
> * Eliminates "garbage in, garbage out" errors from mismatched data vintages.

---
## 📖 Usage Guide

1.  **Upload Data:**
    * On the sidebar (or main page), upload your `Fitch Global Sovereign Data Comparator.xlsb` file.
    * The app will automatically parse the latest period (e.g., "December 2025").

2.  **Select Country:**
    * Choose a sovereign entity (e.g., "Indonesia") from the dropdown.
    * View the **SRM Score**, **Predicted Rating**, and **Actual Rating**.

3.  **Analyze Methodologies:**
    * Navigate to the **"Methodology"** tab to see the specific coefficients used (Structural, Macro, Fiscal, External).

4.  **Run Simulations:**
    * Go to the **"Simulation"** tab.
    * Upload a specific Fitch PDF Report (optional) to see risk warnings.
    * Adjust sliders/inputs for variables like `Govt Debt` or `Real GDP Growth`.
    * Observe the gauge chart update in real-time.

---

## 🧮 Model Overview

The model uses a linear regression approach based on **18 quantitative variables** categorized into four pillars:

| Pillar | Key Indicators |
| :--- | :--- |
| **Structural** | GDP per Capita, Governance (WGI), Share of World GDP, Years Since Default |
| **Macroeconomic** | Real GDP Growth, Inflation, GDP Volatility |
| **Public Finances** | Gross Govt Debt, Interest/Revenue, Fiscal Balance, FC Debt Share |
| **External Finances** | Reserve Currency Status, SNFA, Commodity Dependence, FX Reserves |

*Note: The coefficients used in this dashboard are calibrated to the February 2025 Model Version.*

---

## 📦 Project Structure

```text
sovereign-credit-watch/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .streamlit/
│   └── secrets.toml       # (Optional) For database passcodes
└── assets/                # Images or static files
