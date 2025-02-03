# Stock Market and Fundamental Analysis with ML

**Team:** Team 2  
**Authors:**  
- Busayo Ajayi  
- Promise Sunday   

**Date:** 30.01,2025

---

## Overview

This repository contains the code and analysis for the "Stock Market and Fundamental Analysis with ML" project. The project was developed in collaboration with **Prime INC** to enhance investment decision-making by leveraging machine learning for stock trend prediction and comprehensive fundamental analysis.

---

## Business Introduction

**Company:** Prime INC  
**Industry:** Information Technology, Energy and Utilities

Prime INC is a globally recognized firm known for its strategic investments in emerging markets. Following recent missed investment opportunities due to inefficient decision-making processes, the company partnered with Amdari to adopt data-driven strategies and improve their stock portfolio management.

---

## Business Challenges & Objectives

### Business Challenges
- **Missed Investment Opportunities:** Delays in analyzing stock data led to suboptimal decisions.
- **Lack of Predictive Insight:** Absence of reliable forecasting models.
- **Financial Complexity:** Difficulty interpreting financial statements for long-term profitability.

### Project Objectives
1. **Empower Investors:** Provide actionable insights tailored to novice investors.
2. **Holistic Evaluation:** Merge technical market analysis with fundamental financial assessments.
3. **Enhance Predictive Accuracy:** Utilize machine learning to forecast stock trends.
4. **Mitigate Risks:** Identify potential risks and balance them against anticipated returns.
5. **Enable Data-Driven Decisions:** Offer evidence-based recommendations for investment strategies.

---

## Methodology & Technology Stack

### Project Workflow
1. **Data Investigation & Preparation:**  
   - Define project goals and challenges  
   - Explore data quality, handle missing values/outliers, and perform data transformations

2. **Exploratory Data Analysis (EDA):**  
   - Analyze trends, seasonality, and error components  
   - Visualize daily returns and volatility

3. **Model Building:**  
   - Train forecasting models (ARIMA and ETS)  
   - Experiment with machine learning models (XGBoost and LSTM) for stock direction prediction

4. **Model Evaluation:**  
   - Evaluate model performance using RMSE, MAE, accuracy, precision, recall, and F-score  
   - Conduct cross-validation for robust performance estimates

5. **Forecasting:**  
   - Select the best-fitting model to forecast the next four periods (months)

### Technology Stack
- **Programming Language:** Python
- **Data Analysis & Visualization:**  
  - Numpy, Pandas, Matplotlib, Seaborn
- **Machine Learning:**  
  - Scikit-learn, XGBoost, LSTM (Keras/TensorFlow)
- **Development & Version Control:**  
  - Git, GitHub
- **Interactive Analysis:**  
  - Jupyter Notebook
- **Data Source:**  
  - Yahoo Finance API

---

## Data Overview

- **Stock Selection:**  
  - **Primary Stock:** PG (Procter & Gamble)  
  - **Competitor Stock:** JNJ (Johnson & Johnson)  
  - **Benchmark:** S&P 500 (Americas)

- **Dataset Details:**  
  - **Size:** 3522 rows x 8 columns  
  - **Time Range:** 2010-03-02 to 2024-02-28  
  - **Features:** 'Close', 'High', 'Low', 'Open', 'Volume', 'Return', 'Tomorrow', 'Stocks_Direction'
  - **Quality:** Minimal missing values and no significant outliers
  - **Feature Engineering:** Additional columns such as "Returns", "Tomorrow", and "Stocks-Direction" to capture daily price movements

---

## Exploratory Data Analysis

- **Daily Returns:**  
  - Histograms reveal that the majority of daily returns for PG, JNJ, and the S&P 500 are centered around zero with a few extreme values.

- **Volatility:**  
  - PG shows moderate volatility with a strong correlation (0.97) with the S&P 500.

- **Trading Volume vs. Returns:**  
  - Analysis of volume trends helps assess the predictive power and risk profile of each stock.

---

## Modeling & Evaluation

Two primary models were used to predict stock direction (up or down) for PG:

- **XGBoost:**  
  - Accuracy: 76%  
  - Demonstrated robust predictive power and generalization ability  
- **LSTM:**  
  - Accuracy: 50%  
  - Lower performance compared to XGBoost

**Recommendation:**  
Focus on optimizing and deploying the XGBoost model for stock prediction tasks.

---

## Financial Comparison: PG vs. JNJ

| **Metric**              | **Procter & Gamble (PG)** | **Johnson & Johnson (JNJ)** |
|-------------------------|---------------------------|-----------------------------|
| **Sector**              | Consumer Staples          | Healthcare                  |
| **Market Behavior**     | Defensive, stable growth  | Defensive, strong in downturns |
| **Dividend Yield**      | ~2.5%                     | ~2.8%                       |
| **Total Assets**        | ~$120B                    | ~$187B                      |
| **Net Income**          | ~$14B                     | ~$17B                       |
| **Annual Revenue**      | ~$82B                     | ~$100B                      |
| **P/E Ratio**           | ~25x                      | ~16x                       |

---

## Insights for Investors

- **Return & Risk:**  
  PG and JNJ have similar annualized returns (11%) and volatility (17%), closely tracking the broader market (S&P 500 with 12% return).

- **Risk-Adjusted Performance:**  
  JNJ exhibits a slightly better Sharpe Ratio than PG.

- **Stability & Liquidity:**  
  PG shows lower volatility and higher liquidity, making it a preferable choice for conservative investors.

**Investor Takeaway:**  
- **For Conservative Investors:** PG is recommended due to its stable price movements and consistent dividends.  
- **For Those Seeking Higher Returns:** Consider the broader market (S&P 500) or JNJ based on specific investment strategies.

---

## How to Use This Repository

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo.git](https://github.com/sundaepromix/Stock_Market_Trend_Analysis.git)
