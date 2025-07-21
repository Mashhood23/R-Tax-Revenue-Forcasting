# 📊 R Shiny Tax Forecasting Model Comparison App

## ✨ Overview

This repository hosts an interactive R Shiny application designed to compare and analyze the performance of various machine learning models for forecasting tax collection (`tcny`). The application provides comprehensive data overviews, model accuracy metrics, feature importance insights, and detailed visualizations of actual vs. predicted tax collections, including state-specific analyses and policy recommendations.

The project addresses the critical need for accurate tax revenue forecasting, enabling policymakers to make informed decisions regarding fiscal planning, resource allocation, and economic development strategies. 💰📈

**Link to Live Shiny App: [https://mashhood23.shinyapps.io/TaxForcasting/](https://mashhood23.shinyapps.io/TaxForcasting/)**

## 🧠 Models Implemented

The application compares the following predictive models:

  * **GMM (Generalized Method of Moments):** A robust econometric technique used as a baseline, particularly effective for panel data to address endogeneity and measurement error.

  * **Random Forest:** An ensemble learning method known for its ability to capture complex, non-linear relationships and its robustness against overfitting. It aggregates predictions from multiple decision trees. 🌳🌲

  * **XGBoost:** A highly optimized gradient boosting framework renowned for its speed, flexibility, and state-of-the-art predictive performance. It iteratively builds trees, correcting errors from previous ones. 🚀

  * **LASSO Regression:** A linear model incorporating L1 regularization, which performs automatic feature selection by shrinking less important coefficients to zero, enhancing model interpretability and preventing overfitting in high-dimensional datasets. 📏

## 🌟 Key Features

  * **Data Overview:** Interactive tables and summaries of the raw and transformed data. 📋

  * **Feature Correlation Heatmap:** Visualizes relationships between economic indicators and tax collection. 🔥

  * **State-wise Tax Distribution:** Box plots showing tax collection distribution across different states. 📦

  * **Model Accuracy Metrics:** Displays RMSE, MAE, and MAPE for all models, highlighting the best-performing one. ✅

  * **Hyperparameter Tuning Explanation:** Provides context on how models are optimized. ⚙️

  * **Model Interpretations & Robustness Discussion:** Detailed insights into each model's strengths and suitability for forecasting. 🔍

  * **Interactive Visualizations:** 📊

      * **Actual vs. Predicted Scatter Plots:** Compare overall model performance. 🎯
      * **Aggregated Time Series Plots:** Track national tax collection trends. 📉
      * **State-Specific Time Series Plots:** Analyze actual vs. predicted tax collection for individual states, including Random Forest prediction intervals. 🗺️
      * **Residuals Analysis:** Histograms and scatter plots of prediction errors to assess model bias and variance. 📉
      * **Feature Distribution Analysis:** Explore the distribution of key economic indicators. 📈
      * **State-wise Performance Comparison:** Bar charts showing actual vs. predicted tax collection and residuals by state. 📊

  * **State-Specific Insights & Recommendations:** Dynamically generated policy implications based on model performance for selected states. 💡

  * **Downloadable Results:** Export model metrics and detailed state-wise predictions to CSV. 📥

## 🏗️ Project Structure

```
Tax-Forcasting/
├── app.R                       # Main Shiny application code
├── setup_models.R              # One-time script to train and save models (for local setup)
├── ForecastData.xlsx           # Input data file
└── models/                     # Directory for pre-trained models and data
    ├── rf_model.rds            # Saved Random Forest model
    ├── xgb_model.rds           # Saved XGBoost model
    ├── lasso_model.rds         # Saved LASSO model
    ├── test_data_with_preds.rds# Test data with all model predictions
    └── model_accuracy_metrics.csv # CSV containing model performance metrics
```

## 🚀 Deployment (ShinyApps.io)

This application is designed for easy deployment to [ShinyApps.io](https://www.shinyapps.io/) for a live, interactive demo.

1.  **Create a ShinyApps.io account** if you don't have one.

2.  **Install `rsconnect`** in R: `install.packages("rsconnect")`

3.  **Connect RStudio to your ShinyApps.io account:**

      * In RStudio, go to `Tools` -\> `Global Options` -\> `Publishing` -\> `Connect`.

      * Follow the prompts to paste your token from your ShinyApps.io dashboard.

4.  **Deploy from RStudio:**

      * Open `app.R` in RStudio.

      * Ensure your working directory is set to the project root.

      * Click the "Publish App" button (blue cloud icon with an up arrow) in the top right of the source pane.

      * RStudio will automatically detect `app.R`, `ForecastData.xlsx`, and the entire `models/` directory. Confirm all necessary files are selected.

      * Click "Publish".

## 🤝 Contributing

Feel free to fork this repository, make improvements, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change. 🧑‍💻


## 📧 Contact

  * **Mashhood Raza Khan** - [Your LinkedIn Profile URL Here] (e.g., `https://www.linkedin.com/in/mashhoodrazakhan/`)
  * **Email:** `mashhood1223@gmail.com`
