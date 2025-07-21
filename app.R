# app.R

# ==========================================
# 📦 Load Libraries
# ==========================================
# Ensure these packages are installed. If not, run install.packages("package_name")
# in your R console before running the app.
library(shiny)
library(readxl) # Still needed for initial raw_data load if not pre-processed
library(dplyr)
library(randomForest) # Needed for predict method, not for training
library(xgboost)      # Needed for predict method, not for training
library(glmnet)       # Needed for predict method, not for training
library(ggplot2)
library(DT) # For interactive tables
library(extrafont) # For better font handling in plots (run font_import() once)
library(Cairo)     # For high-quality plot output, especially with special characters
library(shinythemes) # For modern UI themes
library(tidyr)     # For pivot_longer in time series plot
library(caret)     # For varImp function (if models were trained with caret, but we'll load raw models now)
library(corrplot)  # For correlation heatmap
library(plotly)    # For interactive visualizations
library(e1071)     # For skewness calculation in feature distribution interpretation
library(rlang)     # For !!sym() in ggplot aesthetics

# --- One-time Font Setup (Run in R Console, NOT in Shiny app.R) ---
# If you haven't done this before, run these lines in your R CONSOLE:
# install.packages(c("extrafont", "Cairo", "shinythemes", "tidyr", "caret", "corrplot", "plotly", "e1071", "rlang"))
# library(extrafont)
# font_import(prompt = FALSE) # This can take a few minutes.
# loadfonts(device = "win")   # For Windows users.
# ------------------------------------------------------------------

# Load fonts for the Shiny session (will only work if font_import() was run previously)
tryCatch({
  loadfonts(device = "win") # For Windows
}, error = function(e) {
  message("Failed to load fonts for Windows device. Plots might have character issues if font_import() was not run previously.")
})

# Define the directory where models are saved
model_dir <- "models" # This is a relative path within your app's directory

# ==========================================
# UI (User Interface)
# ==========================================
ui <- fluidPage(
  # Apply a modern Shiny theme
  theme = shinytheme("flatly"), # You can try "cosmo", "cerulean", "simplex", etc.
  
  titlePanel("Tax Forecasting Model Comparison App"), # Removed name from title
  
  navbarPage(
    "Navigation",
    tabPanel("Data Overview", icon = icon("table"),
             sidebarLayout(
               sidebarPanel(
                 h4("Data Source"),
                 helpText("The 'ForecastData.xlsx' file is pre-loaded from the app's directory.")
               ),
               mainPanel(
                 h3("Raw Data Head (First 5 Rows)"),
                 DTOutput("data_head"),
                 h3("Data Summary"),
                 verbatimTextOutput("data_summary"),
                 tags$hr(),
                 h3("Feature Correlation Heatmap"),
                 plotOutput("correlation_heatmap", height = "600px"),
                 htmlOutput("correlation_heatmap_interpretation"),
                 tags$hr(),
                 h3("Tax Collection Distribution by State"),
                 plotOutput("state_boxplot", height = "600px"),
                 htmlOutput("state_boxplot_interpretation")
               )
             )
    ),
    tabPanel("Model Training & Metrics", icon = icon("brain"),
             sidebarLayout(
               sidebarPanel(
                 helpText("Models are pre-trained and loaded automatically on app startup for faster performance. Please ensure you've run the 'One-Time Model Training and Saving Script' first.")
               ),
               mainPanel(
                 h3("Model Accuracy Metrics"),
                 DTOutput("metrics_table"),
                 downloadButton("downloadMetrics", "Download Metrics CSV", class = "btn-success"),
                 tags$hr(),
                 h3("Understanding Hyperparameter Tuning"),
                 htmlOutput("tuning_explanation"), # New explanation for tuning
                 tags$hr(),
                 h3("Model Interpretations"),
                 htmlOutput("model_interpretations"), # Pre-defined interpretation
                 tags$hr(),
                 h3("Model Robustness Discussion"), # Pre-defined robustness discussion
                 htmlOutput("model_robustness_discussion")
               )
             )
    ),
    tabPanel("Results & Visualizations", icon = icon("chart-line"),
             sidebarLayout(
               sidebarPanel(
                 selectInput("plot_model_select", "Select Model for Residuals/Time Series:",
                             choices = c("Random Forest" = "rf_pred",
                                         "XGBoost" = "xgb_pred",
                                         "LASSO" = "lasso_pred",
                                         "GMM (Baseline)" = "gmm_predicted"),
                             selected = "rf_pred"),
                 uiOutput("state_selector"), # Dynamic selector for states
                 tags$hr(),
                 h4("Results Table Options"),
                 downloadButton("downloadResults", "Download Results CSV", class = "btn-success")
               ),
               mainPanel(
                 fluidRow(
                   column(12, h3("Actual vs Predicted Scatter Plot (Interactive)")),
                   column(12, plotlyOutput("scatter_plot", height = "500px"))
                 ),
                 fluidRow(
                   column(12, h3("Aggregated Actual vs Predicted Time Series (Interactive)")),
                   column(12, plotlyOutput("aggregated_time_series", height = "500px")),
                   column(12, htmlOutput("aggregated_time_series_interpretation"))
                 ),
                 fluidRow(
                   column(6, h3("Random Forest Feature Importance")),
                   column(6, h3("XGBoost Feature Importance"))
                 ),
                 fluidRow(
                   column(6, plotOutput("rf_importance_plot", height = "500px")),
                   column(6, plotOutput("xgb_importance_plot", height = "500px"))
                 ),
                 fluidRow(
                   column(12, h3("Time Series Plot (Actual vs. Predicted for Selected State) (Interactive)")),
                   column(12, htmlOutput("ci_explanation")), # Explanation for CI
                   column(12, plotlyOutput("time_series_plot", height = "500px"))
                 ),
                 fluidRow(
                   column(6, h3("Residuals Distribution (Histogram) (Interactive)")),
                   column(6, h3("Residuals vs. Predicted Plot (Interactive)"))
                 ),
                 fluidRow(
                   column(6, plotlyOutput("residuals_hist_plot", height = "400px")),
                   column(6, plotlyOutput("residuals_vs_predicted_plot", height = "400px"))
                 ),
                 fluidRow(
                   column(12, h3("State-Specific Insights & Recommendations")),
                   htmlOutput("state_recommendations") # Pre-defined recommendation
                 ),
                 fluidRow(
                   column(12, h3("State-Wise Results Table (Head)")),
                   DTOutput("results_table")
                 )
               )
             )
    ),
    # NEW: Detailed Visualizations Tab
    tabPanel("Detailed Visualizations", icon = icon("chart-bar"),
             sidebarLayout(
               sidebarPanel(
                 h4("Feature Distribution Analysis"),
                 selectInput("dist_feature_select", "Select Feature to Visualize:",
                             choices = c(
                               "Tax Collection (TCNY)" = "tcny",
                               "Per Capita EPF Subscribers" = "pcEPF",
                               "Total UPI Users" = "UPIu",
                               "Service Sector GSVA" = "serviceGSVAr",
                               "Per Capita Bank Deposits" = "pcbd",
                               "Literacy Rate (LRR)" = "lrr",
                               "Urbanization Rate (URR)" = "urr"
                             ),
                             selected = "tcny"),
                 tags$hr(),
                 h4("State-wise Performance Comparison"),
                 selectInput("state_comparison_model", "Select Model for State Comparison:",
                             choices = c("Random Forest" = "rf_pred",
                                         "XGBoost" = "xgb_pred",
                                         "LASSO" = "lasso_pred",
                                         "GMM (Baseline)" = "gmm_predicted"),
                             selected = "rf_pred")
               ),
               mainPanel(
                 h3("Distribution of Selected Feature (Interactive)"),
                 plotlyOutput("feature_distribution_plot", height = "500px"),
                 htmlOutput("feature_distribution_interpretation"),
                 tags$hr(),
                 h3("Actual vs. Predicted Tax Collection by State (Interactive)"),
                 plotlyOutput("actual_vs_predicted_state_plot", height = "600px"),
                 htmlOutput("actual_vs_predicted_state_interpretation"),
                 tags$hr(),
                 h3("Prediction Errors (Residuals) by State (Interactive)"),
                 plotlyOutput("residuals_by_state_plot", height = "600px"),
                 htmlOutput("residuals_by_state_interpretation"),
                 tags$hr(),
                 # NEW: State Prediction Accuracy Patterns Section
                 h3("State Prediction Accuracy Patterns"),
                 htmlOutput("accuracy_pattern_interpretation"),
                 DTOutput("state_accuracy_table")
               )
             )
    )
  ),
  tags$footer(
    HTML("<hr><p style='text-align: center; font-size: 0.9em; color: gray;'>Tax Forecasting Model Analysis</p>") # Removed name from footer
  )
)

# ==========================================
# Server Logic
# ==========================================
server <- function(input, output, session) {
  
  # Reactive expression to read the pre-loaded Excel file
  # This is still needed for the initial data overview and transformations
  raw_data <- reactive({
    file_path <- "ForecastData.xlsx"
    if (!file.exists(file_path)) {
      showNotification(
        paste("Error: 'ForecastData.xlsx' not found in the app's directory. Please place it next to app.R."),
        type = "error", duration = NULL, closeButton = TRUE
      )
      return(NULL)
    }
    read_excel(file_path)
  })
  
  # Reactive expression for data transformation
  # This is still needed for data overview tab and to ensure consistency
  transformed_data <- reactive({
    data <- raw_data()
    req(data)
    
    required_cols_for_transform <- c("pcEPF", "UPIu", "serviceGSVAr", "pcbd", "lrr", "urr", "tcny", "year", "states", "gmm_predicted")
    if (!all(required_cols_for_transform %in% colnames(data))) {
      showNotification(paste("Missing one or more required columns for transformation:",
                             paste(setdiff(required_cols_for_transform, colnames(data)), collapse = ", ")),
                       type = "error", duration = 10)
      return(NULL)
    }
    
    data %>%
      mutate(
        log_pcepf    = log(pcEPF),
        log_upiu     = log(UPIu),
        log_service  = log(serviceGSVAr),
        log_pcbd     = log(pcbd),
        log_lrrurr   = log(lrr * urr),
        tax_lag      = lag(tcny) # TCNY(-1)
      ) %>%
      filter(!is.na(tax_lag))
  })
  
  # Render raw data head (first 5 rows)
  output$data_head <- renderDT({
    req(transformed_data())
    datatable(head(transformed_data(), 5),
              options = list(pageLength = 5, scrollX = TRUE, dom = 'tip'),
              caption = "First 5 rows of the processed data")
  })
  
  # Render raw data summary
  output$data_summary <- renderPrint({
    req(transformed_data())
    summary(transformed_data())
  })
  
  # Correlation Heatmap
  output$correlation_heatmap <- renderPlot({
    req(transformed_data())
    data_for_cor <- transformed_data() %>%
      select(log_pcepf, log_upiu, log_service, log_pcbd, log_lrrurr, tax_lag, tcny) %>%
      na.omit()
    
    if (ncol(data_for_cor) < 2) {
      return(NULL)
    }
    
    corrplot(cor(data_for_cor), method = "color", type = "upper",
             tl.col = "black", tl.srt = 45,
             addCoef.col = "black",
             number.cex = 0.8,
             mar = c(0,0,1,0),
             title = "Correlation Matrix of Key Variables")
  }, res = 96)
  
  # Correlation Heatmap Interpretation
  output$correlation_heatmap_interpretation <- renderUI({
    req(transformed_data())
    HTML(
      "<h4>Understanding Feature Relationships:</h4>",
      "<p>This heatmap visually represents the strength and direction of relationships between various economic indicators (features) and the 'Tax Collection (tcny)' variable, as well as among the features themselves. This helps this study understand which factors move together.</p>",
      "<ul>",
      "<li><b>Colors and Numbers:</b> Red shades indicate a strong positive relationship (as one factor increases, the other tends to increase), while blue shades indicate a strong negative relationship (as one factor increases, the other tends to decrease). The numbers within each square, known as correlation coefficients, quantify this relationship, ranging from -1 (perfect negative) to +1 (perfect positive). Values near 0 suggest a weak or no linear relationship.</li>",
      "<li><b>Insights for Policymakers:</b>",
      "<ul>",
      "<li><b>Predictive Power:</b> Focus on the correlations with 'tcny' (the bottom row). Features with high absolute correlation values (closer to +1 or -1) are strong indicators for tax collection. For example, if 'log_pcepf' shows a high positive correlation with 'tcny', it suggests that robust growth in Employee Provident Fund contributions is strongly associated with higher tax revenue, indicating a healthy formal sector economy.</li>",
      "<li><b>Interdependencies:</b> Examining correlations among predictor variables can reveal underlying economic structures. High correlations between two distinct economic indicators might imply they are influenced by common macroeconomic forces or have a causal link. Understanding these interdependencies is crucial for designing holistic economic policies, as interventions targeting one factor might have ripple effects on others.</li>",
      "</ul>",
      "</li>",
      "</ul>"
    )
  })
  
  # Box Plot of Tax Collection by State
  output$state_boxplot <- renderPlot({
    req(transformed_data())
    plot_data <- transformed_data() %>%
      mutate(states = factor(states))
    
    ggplot(plot_data, aes(x = states, y = tcny, fill = states)) +
      geom_boxplot(alpha = 0.7) +
      labs(title = "Distribution of Tax Collection (₹ Crore) by State (2018-2022)",
           x = "State",
           y = "Tax Collection (₹ Crore)") +
      theme_minimal() +
      theme(text = element_text(family = "sans", size = 12),
            axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
            plot.title = element_text(hjust = 0.5, face = "bold"),
            legend.position = "none") +
      scale_fill_viridis_d()
  }, res = 96)
  
  # Box Plot Interpretation
  output$state_boxplot_interpretation <- renderUI({
    req(transformed_data())
    HTML(
      "<h4>Understanding State-wise Tax Performance:</h4>",
      "<p>This box plot visualizes the distribution of tax collection (TCNY) for each state over the period 2018-2022. Each 'box' represents the middle 50% of tax collection values for that state, with the central line indicating the median (typical) tax collection. This visualization helps this study identify variations in state performance.</p>",
      "<ul>",
      "<li><b>Comparing States:</b> By observing the position of each box, you can quickly identify states with generally higher or lower tax collection levels based on the position of their boxes. States with boxes positioned higher on the y-axis typically demonstrate stronger revenue generation.</li>",
      "<li><b>Variability and Consistency:</b> The height of each box illustrates the consistency of a state's tax collection over the period. A taller box suggests greater variability in annual tax collection, while a shorter box indicates more stable and predictable revenue. The 'whiskers' extend to show the full range of typical data, and individual points outside the whiskers represent potential outliers—years with unusually high or low collection that warrant specific investigation.</li>",
      "<li><b>Policy Implication:</b> This visualization provides critical insights for policymakers. States with consistently high and stable tax collection (shorter, higher boxes) might offer 'best practices' in economic policy or tax administration that could be emulated. Conversely, states exhibiting low average collection or high variability might require targeted economic development initiatives, a.review of their tax policies, or an enhancement of their tax administration strategies to stabilize and boost revenue. Understanding these patterns is fundamental for equitable resource allocation and development planning."
    )
  })
  
  # Reactive for model results (now loads pre-trained data)
  model_results <- reactive({
    # Paths to saved models and data
    rf_model_path <- file.path(model_dir, "rf_model.rds")
    xgb_model_path <- file.path(model_dir, "xgb_model.rds")
    lasso_model_path <- file.path(model_dir, "lasso_model.rds")
    test_data_path <- file.path(model_dir, "test_data_with_preds.rds")
    metrics_table_path <- file.path(model_dir, "model_accuracy_metrics.csv")
    
    # Check if pre-trained files exist
    if (!file.exists(rf_model_path) || !file.exists(xgb_model_path) ||
        !file.exists(lasso_model_path) || !file.exists(test_data_path) ||
        !file.exists(metrics_table_path)) {
      showNotification(
        paste("Error: One or more pre-trained model/data files not found in the '", model_dir,
              "' directory. Please run the 'One-Time Model Training and Saving Script' first."),
        type = "error", duration = NULL, closeButton = TRUE
      )
      return(NULL) # Return NULL if files are not found
    }
    
    withProgress(message = 'Loading pre-trained models and data...', value = 0, {
      incProgress(0.2, detail = "Loading Random Forest model...")
      rf_model <- readRDS(rf_model_path)
      
      incProgress(0.4, detail = "Loading XGBoost model...")
      xgb_model <- readRDS(xgb_model_path)
      
      incProgress(0.6, detail = "Loading LASSO model...")
      lasso_model <- readRDS(lasso_model_path)
      
      incProgress(0.8, detail = "Loading test data with predictions...")
      test_data_with_preds <- readRDS(test_data_path)
      
      incProgress(0.9, detail = "Loading accuracy metrics...")
      metrics_table <- read.csv(metrics_table_path, row.names = 1)
      
      incProgress(1, detail = "Done!")
      
      list(
        rf_model = rf_model,
        xgb_model = xgb_model,
        lasso_model = lasso_model,
        test_data_with_preds = test_data_with_preds,
        metrics_table = metrics_table
      )
    })
  })
  
  # Render Metrics Table
  output$metrics_table <- renderDT({
    req(model_results())
    datatable(as.data.frame(model_results()$metrics_table),
              options = list(pageLength = 5, dom = 'tip'),
              rownames = TRUE,
              caption = "Model Accuracy Metrics")
  })
  
  # Download Metrics CSV
  output$downloadMetrics <- downloadHandler(
    filename = function() {
      "model_accuracy_metrics.csv"
    },
    content = function(file) {
      req(model_results())
      write.csv(as.data.frame(model_results()$metrics_table), file, row.names = TRUE)
    }
  )
  
  # New: Explanation for Hyperparameter Tuning
  output$tuning_explanation <- renderUI({
    HTML(
      "<h4>What is Hyperparameter Tuning?</h4>",
      "<p>Think of a model like a complex machine with many knobs and dials. These knobs are called <b>hyperparameters</b> (e.g., how many 'trees' in a forest, or how 'aggressively' a model learns). The default settings for these knobs might not be the best for every dataset.</p>",
      "<p><b>Hyperparameter tuning</b> is the process of systematically trying different combinations of these knob settings to find the ones that make the model perform its absolute best for your specific data. It's like finding the 'sweet spot' for your machine to run most efficiently and accurately.</p>",
      "<p>In this app, the models were pre-tuned and trained to optimize their performance on the tax data, aiming to reduce the differences between predicted and actual tax collections. This pre-training makes the app load much faster.</p>"
    )
  })
  
  
  # Static Model Interpretations (Pre-defined Text Block)
  output$model_interpretations <- renderUI({
    req(model_results())
    metrics <- model_results()$metrics_table
    best_model_rmse <- rownames(metrics)[which.min(metrics[, "RMSE"])]
    best_model_mae <- rownames(metrics)[which.min(metrics[, "MAE"])]
    best_model_mape <- rownames(metrics)[which.min(metrics[, "MAPE"])]
    
    HTML(paste0(
      "<h4>General Model Insights (by this Study):</h4>",
      "<p>This analysis employs four distinct modeling approaches to forecast tax collection (TCNY), each offering unique strengths:</p>",
      "<ul>",
      "<li><b>GMM (Generalized Method of Moments):</b> A robust econometric technique often used to address endogeneity and measurement error, providing a foundational benchmark, particularly in panel data settings.</li>",
      "<li><b>Random Forest:</b> An ensemble learning method that aggregates predictions from multiple decision trees. It excels at capturing complex, non-linear relationships and is highly resistant to overfitting, making it a powerful tool for diverse datasets.</li>",
      "<li><b>XGBoost:</b> A highly optimized gradient boosting framework. It's known for its speed, flexibility, and predictive power. It iteratively builds decision trees, with each new tree correcting the errors of the preceding ones, leading to state-of-the-art performance in many predictive tasks.</li>",
      "<li><b>LASSO Regression:</b> A linear model that incorporates L1 regularization, simultaneously performing variable selection and shrinkage. This makes it particularly effective in high-dimensional datasets by driving less important feature coefficients to zero, thus enhancing model interpretability and preventing overfitting.</li>",
      "</ul>",
      "<h4>Interpreting Model Performance Metrics:</h4>",
      "<ul>",
      "<li><b>RMSE (Root Mean Squared Error):</b> Quantifies the average magnitude of the errors, giving more weight to larger errors. It is expressed in the same units as your tax collection (₹ Crore), making it directly interpretable as the typical deviation of predictions from actual values. A lower RMSE indicates better predictive accuracy.</li>",
      "<li><b>MAE (Mean Absolute Error):</b> Measures the average magnitude of the errors without considering their direction. Like RMSE, it's expressed in ₹ Crore and provides a straightforward average error. MAE is less sensitive to outliers compared to RMSE. A lower MAE indicates better average accuracy.</li>",
      "<li><b>MAPE (Mean Absolute Percentage Error):</b> Expresses the error as a percentage of the actual value. This metric is useful for comparing accuracy across different scales of tax collection or different states, as it's scale-independent. However, it can be unstable or infinite if actual values are zero or very close to zero. A lower MAPE indicates better percentage accuracy.</li>",
      "</ul>",
      "<p><b>Overall Performance:</b> Based on the calculated metrics, this study suggests that the model with the lowest RMSE is <b>", best_model_rmse, "</b>, the model with the lowest MAE is <b>", best_model_mae, "</b>, and the model with the lowest MAPE is <b>", best_model_mape, "</b>. When selecting the 'best' model, consider which error metric aligns most closely with your policy objectives. For instance, if minimizing the impact of large forecasting errors is paramount, RMSE might be your primary guide. If a consistent average error across all predictions is preferred, MAE would be more relevant.</p>"
    ))
  })
  
  # Static Model Robustness Discussion (Pre-defined Text Block)
  output$model_robustness_discussion <- renderUI({
    req(model_results())
    HTML(paste0(
      "<h4>Model Robustness Discussion (by this Study):</h4>",
      "<p>The robustness of a forecasting model refers to its ability to perform consistently well even when faced with variations or imperfections in the data, or when applied to new, unseen data. This study evaluates the following:</p>",
      "<ul>",
      "<li><b>GMM (Generalized Method of Moments):</b> A robust econometric technique often used to address endogeneity and measurement error, providing a foundational benchmark, particularly in panel data settings.</li>",
      "<li><b>Random Forest:</b> Highly robust to outliers and noise in the data due to its ensemble nature (averaging multiple trees). It handles non-linear relationships and multicollinearity (highly correlated features) very well without requiring explicit feature scaling. Its robustness comes from diversity in trees and bagging, reducing overfitting. It's generally stable across different datasets, but can be computationally intensive for very large datasets.</li>",
      "<li><b>XGBoost:</b> Known for its exceptional performance and efficiency. It is robust to outliers to some extent due to its gradient boosting mechanism, which focuses on correcting errors. It handles non-linearity and multicollinearity effectively. Its robustness also stems from regularization techniques (L1/L2) that prevent overfitting. XGBoost is often considered very robust for predictive accuracy but requires careful tuning of hyperparameters for optimal performance.</li>",
      "<li><b>LASSO Regression:</b> While a linear model, LASSO introduces L1 regularization, which makes it robust in terms of feature selection by shrinking less important coefficients to zero. This helps prevent overfitting in high-dimensional data and can make the model more interpretable. However, its linearity means it might not capture complex non-linear relationships as well as tree-based models, and its performance can be sensitive to scaling of features. It is less robust to multicollinearity if multiple features are highly correlated, as it might arbitrarily pick one.</li>",
      "</ul>",
      "<p>In summary, while all models have their strengths, Random Forest and XGBoost generally offer higher robustness to complex data patterns and outliers, making them highly reliable for tax forecasting. LASSO provides robustness through feature selection, while GMM offers econometric stability, particularly for panel data structures. The choice depends on the specific data characteristics and the desired trade-off between interpretability and predictive power, as highlighted by this study.</p>"
    ))
  })
  
  
  # Render Results Table
  output$results_table <- renderDT({
    req(model_results())
    # The results_table is now part of test_data_with_preds, which is loaded
    results_table_from_loaded <- model_results()$test_data_with_preds %>%
      select(states, tcny, gmm_predicted, rf_pred, xgb_pred, lasso_pred,
             rf_lower_90, rf_upper_90, rf_lower_95, rf_upper_95) %>% # Include CI columns
      mutate(
        residual_gmm    = tcny - gmm_predicted,
        residual_rf     = tcny - rf_pred,
        residual_xgb    = tcny - xgb_pred,
        residual_lasso  = tcny - lasso_pred
      )
    datatable(head(results_table_from_loaded),
              options = list(pageLength = 5, scrollX = TRUE),
              caption = "State-Wise Results (Head)")
  })
  
  # Download Results CSV
  output$downloadResults <- downloadHandler(
    filename = function() {
      "tax_forecast_results.csv"
    },
    content = function(file) {
      req(model_results())
      results_table_full <- model_results()$test_data_with_preds %>%
        select(states, tcny, gmm_predicted, rf_pred, xgb_pred, lasso_pred,
               rf_lower_90, rf_upper_90, rf_lower_95, rf_upper_95) %>% # Include CI columns
        mutate(
          residual_gmm    = tcny - gmm_predicted,
          residual_rf     = tcny - rf_pred,
          residual_xgb    = tcny - xgb_pred,
          residual_lasso  = tcny - lasso_pred
        )
      write.csv(results_table_full, file, row.names = FALSE)
    }
  )
  
  # Render Actual vs Predicted Scatter Plot (Interactive with Plotly)
  output$scatter_plot <- renderPlotly({
    req(model_results())
    results_table_full <- model_results()$test_data_with_preds
    
    p <- ggplot(results_table_full, aes(x = tcny)) +
      geom_point(aes(y = gmm_predicted, color = "GMM", text = paste("State:", states, "<br>Actual:", tcny, "<br>Predicted:", gmm_predicted)), size = 3, alpha = 0.7) +
      geom_point(aes(y = rf_pred, color = "Random Forest", text = paste("State:", states, "<br>Actual:", tcny, "<br>Predicted:", rf_pred)), size = 3, alpha = 0.7) +
      geom_point(aes(y = xgb_pred, color = "XGBoost", text = paste("State:", states, "<br>Actual:", tcny, "<br>Predicted:", xgb_pred)), size = 3, alpha = 0.7) +
      geom_point(aes(y = lasso_pred, color = "LASSO", text = paste("State:", states, "<br>Actual:", tcny, "<br>Predicted:", lasso_pred)), size = 3, alpha = 0.7) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black", size = 1) +
      labs(title = "Actual vs Predicted Tax (2023)",
           x = "Actual Tax Collection (₹ Crore)",
           y = "Predicted Tax Collection (₹ Crore)",
           color = "Model") +
      theme_minimal() +
      theme(text = element_text(family = "sans", size = 14),
            plot.title = element_text(hjust = 0.5, face = "bold"),
            legend.position = "bottom") +
      scale_color_brewer(palette = "Set1")
    
    ggplotly(p, tooltip = "text") %>%
      layout(hovermode = "closest")
  })
  
  # Render Random Forest Feature Importance Plot (Now uses loaded model)
  output$rf_importance_plot <- renderPlot({
    req(model_results()$rf_model)
    rf_model <- model_results()$rf_model
    # For randomForest object, use varImpPlot directly
    par(mar = c(5, 8, 4, 2) + 0.1) # Adjust margins for labels
    varImpPlot(rf_model, main = "Random Forest Feature Importance", cex.names = 1.2)
    par(mar = c(5, 4, 4, 2) + 0.1) # Reset margins
  }, res = 96)
  
  # Render XGBoost Feature Importance Plot (Now uses loaded model)
  output$xgb_importance_plot <- renderPlot({
    req(model_results()$xgb_model)
    xgb_model <- model_results()$xgb_model
    # For xgboost object, use xgb.importance and xgb.plot.importance
    xgb_imp <- xgb.importance(model = xgb_model)
    par(mar = c(5, 8, 4, 2) + 0.1) # Adjust margins for labels
    xgb.plot.importance(xgb_imp, main = "XGBoost Feature Importance", cex = 1.2)
    par(mar = c(5, 4, 4, 2) + 0.1) # Reset margins
  }, res = 96)
  
  # Aggregated Time Series Plot (Interactive with Plotly)
  output$aggregated_time_series <- renderPlotly({
    req(model_results())
    aggregated_data <- model_results()$test_data_with_preds %>%
      group_by(year) %>%
      summarise(
        Actual = sum(tcny, na.rm = TRUE),
        GMM_Predicted = sum(gmm_predicted, na.rm = TRUE),
        RF_Predicted = sum(rf_pred, na.rm = TRUE),
        XGB_Predicted = sum(xgb_pred, na.rm = TRUE),
        LASSO_Predicted = sum(lasso_pred, na.rm = TRUE)
      ) %>%
      tidyr::pivot_longer(cols = -year, names_to = "Model", values_to = "Value")
    
    p <- ggplot(aggregated_data, aes(x = year, y = Value, color = Model, group = Model,
                                     text = paste("Year:", year, "<br>Model:", Model, "<br>Value:", round(Value, 2), "Cr"))) +
      geom_line(size = 1.2) +
      geom_point(size = 3) +
      labs(title = "Aggregated Actual vs Predicted Tax Collection Over Time",
           x = "Year",
           y = "Total Tax Collection (₹ Crore)",
           color = "Model") +
      theme_minimal() +
      theme(text = element_text(family = "sans", size = 14),
            plot.title = element_text(hjust = 0.5, face = "bold"),
            legend.position = "bottom") +
      scale_color_brewer(palette = "Paired") +
      scale_x_continuous(breaks = unique(aggregated_data$year))
    
    ggplotly(p, tooltip = "text") %>%
      layout(hovermode = "x unified")
  })
  
  # Aggregated Time Series Interpretation
  output$aggregated_time_series_interpretation <- renderUI({
    req(model_results())
    HTML(
      "<h4>Overall Trend Analysis:</h4>",
      "<p>This plot illustrates the total tax collection across all states, comparing the actual values with the aggregated predictions from each model over time. It provides a macro-level perspective on how effectively this study's models capture the overarching trend in national tax revenue.</p>",
      "<ul>",
      "<li><b>Trend Alignment:</b> Observe how closely the predicted lines align with the 'Actual' line. Strong alignment suggests that our models are effective at forecasting the general direction and magnitude of national tax revenue, indicating a reliable macro-level predictive capability.</li>",
      "<li><b>Periods of Deviation:</b> Any significant discrepancies between the 'Actual' line and the predicted lines highlight years where the models, on average, either over- or underestimated total tax collection. Such deviations warrant deeper investigation into significant national economic events, major policy shifts, or unforeseen external factors that occurred during those periods.</li>",
      "<li><b>Policy Implication:</b> This aggregated view is indispensable for national fiscal planning and budget formulation. If our models consistently predict lower than what this study suggests is actual, it might indicate an opportunity to set more ambitious revenue targets. Conversely, consistent overprediction could lead to unexpected budget shortfalls. Understanding these trends helps policymakers set realistic national budget expectations, assess the overall health and responsiveness of the tax system, and plan for future economic stability.</li>",
      "</ul>"
    )
  })
  
  
  # Dynamic State Selector for Time Series Plot
  output$state_selector <- renderUI({
    req(model_results())
    states_list <- sort(unique(model_results()$test_data_with_preds$states))
    selectInput("selected_state", "Select State for Time Series:",
                choices = states_list,
                selected = states_list[1])
  })
  
  # NEW: Explanation for Confidence Intervals
  output$ci_explanation <- renderUI({
    req(input$plot_model_select)
    if (input$plot_model_select == "rf_pred") {
      HTML(
        "<h4>Understanding Prediction Intervals (Random Forest):</h4>",
        "<p>For the Random Forest model, the vertical bars around the prediction point represent <b>prediction intervals</b>. These intervals give you a range within which the model is confident the actual tax collection will fall.</p>",
        "<ul>",
        "<li>The <b>darker bar (90% CI)</b> means that based on the model, there's a 90% chance the actual tax collection will be within this range.</li>",
        "<li>The <b>lighter bar (95% CI)</b> is a wider range, indicating a 95% chance that the actual tax collection will fall within it.</li>",
        "</ul>",
        "<p>A shorter bar suggests higher certainty in the model's prediction for that specific state and year. If the actual tax collection falls outside these bars, it indicates an unexpected deviation that the model did not anticipate with that level of confidence.</p>"
      )
    } else if (input$plot_model_select == "xgb_pred") {
      HTML(
        "<h4>Note on Prediction Intervals for XGBoost:</h4>",
        "<p>XGBoost primarily provides point predictions (a single best estimate). Generating reliable prediction intervals for XGBoost is more complex and typically requires advanced techniques like bootstrapping or specialized quantile regression methods, which are beyond the scope of this app's current implementation.</p>"
      )
    } else {
      HTML(
        "<h4>Note on Prediction Intervals:</h4>",
        "<p>Prediction intervals are not currently available for the selected model. They are most readily available for Random Forest in this application.</p>"
      )
    }
  })
  
  # Render Time Series Plot (Interactive with Plotly)
  output$time_series_plot <- renderPlotly({
    req(model_results(), input$selected_state)
    plot_data_filtered_state <- model_results()$test_data_with_preds %>%
      filter(states == input$selected_state)
    
    if (nrow(plot_data_filtered_state) == 0) {
      return(NULL)
    }
    
    line_plot_data <- plot_data_filtered_state %>%
      select(year, tcny, gmm_predicted, rf_pred, xgb_pred, lasso_pred) %>%
      tidyr::pivot_longer(cols = -year, names_to = "Model", values_to = "Value")
    
    
    p <- ggplot(line_plot_data, aes(x = year, y = Value, color = Model, group = Model,
                                    text = paste("Year:", year, "<br>Model:", Model, "<br>Value:", round(Value, 2), "Cr"))) +
      geom_line(size = 1.2) +
      geom_point(size = 3) +
      labs(title = paste("Actual vs Predicted Tax Over Time for", input$selected_state),
           x = "Year",
           y = "Tax Collection (₹ Crore)",
           color = "Model") +
      theme_minimal() +
      theme(text = element_text(family = "sans", size = 14),
            plot.title = element_text(hjust = 0.5, face = "bold"),
            legend.position = "bottom") +
      scale_x_continuous(breaks = unique(line_plot_data$year))
    
    if (input$plot_model_select == "rf_pred") {
      rf_ci_data_for_plot <- plot_data_filtered_state
      
      if (nrow(rf_ci_data_for_plot) > 0 &&
          all(c("year", "rf_lower_95", "rf_upper_95", "rf_lower_90", "rf_upper_90") %in% colnames(rf_ci_data_for_plot))) {
        p <- p +
          geom_errorbar(data = rf_ci_data_for_plot,
                        aes(x = year, ymin = rf_lower_95, ymax = rf_upper_95, color = "RF 95% CI"),
                        width = 0.15, size = 0.8, inherit.aes = FALSE) +
          geom_errorbar(data = rf_ci_data_for_plot,
                        aes(x = year, ymin = rf_lower_90, ymax = rf_upper_90, color = "RF 90% CI"),
                        width = 0.1, size = 1, inherit.aes = FALSE) +
          scale_color_manual(name = "Legend",
                             values = c(
                               "tcny" = "#1B9E77", # Actual
                               "gmm_predicted" = "#D95F02", # GMM
                               "rf_pred" = "#7570B3", # Random Forest
                               "xgb_pred" = "#E7298A", # XGBoost
                               "lasso_pred" = "#66A61E", # LASSO
                               "RF 90% CI" = "darkblue",
                               "RF 95% CI" = "steelblue"
                             ),
                             labels = c(
                               "tcny" = "Actual Tax",
                               "gmm_predicted" = "GMM Predicted",
                               "rf_pred" = "Random Forest Predicted",
                               "xgb_pred" = "XGBoost Predicted",
                               "lasso_pred" = "LASSO Predicted",
                               "RF 90% CI" = "RF 90% CI",
                               "RF 95% CI" = "RF 95% CI"
                             ),
                             guide = guide_legend(override.aes = list(
                               linetype = c("solid", "solid", "solid", "solid", "solid", "solid", "solid"),
                               shape = c(16, 16, 16, 16, 16, NA, NA),
                               size = c(3, 3, 3, 3, 3, 1, 0.8)
                             )))
      } else {
        p <- p + scale_color_brewer(palette = "Dark2")
      }
    } else {
      p <- p + scale_color_brewer(palette = "Dark2")
    }
    
    ggplotly(p, tooltip = "text") %>%
      layout(hovermode = "x unified")
  })
  
  # Render Residuals Histogram
  output$residuals_hist_plot <- renderPlotly({
    req(model_results(), input$plot_model_select)
    results_data <- model_results()$test_data_with_preds
    
    selected_pred_col <- input$plot_model_select
    results_data$residuals <- results_data$tcny - results_data[[selected_pred_col]]
    
    p <- ggplot(results_data, aes(x = residuals, text = paste("Residual:", round(residuals, 2), "Cr"))) +
      geom_histogram(binwidth = diff(range(results_data$residuals, na.rm = TRUE))/30, fill = "steelblue", color = "black", alpha = 0.7) +
      geom_density(aes(y = after_stat(density) * diff(range(results_data$residuals, na.rm = TRUE))/30), color = "darkblue", size = 1) +
      labs(title = paste("Distribution of Residuals for", names(input$plot_model_select)[input$plot_model_select == selected_pred_col]),
           x = "Residuals (This Study's Actual - Predicted)",
           y = "Frequency") +
      theme_minimal() +
      theme(text = element_text(family = "sans", size = 14),
            plot.title = element_text(hjust = 0.5, face = "bold"))
    
    ggplotly(p, tooltip = "text")
  })
  
  # Render Residuals vs. Predicted Plot
  output$residuals_vs_predicted_plot <- renderPlotly({
    req(model_results(), input$plot_model_select)
    results_data <- model_results()$test_data_with_preds
    
    selected_pred_col <- input$plot_model_select
    results_data$predicted_values <- results_data[[selected_pred_col]]
    results_data$residuals <- results_data$tcny - results_data[[selected_pred_col]]
    
    
    p <- ggplot(results_data, aes(x = predicted_values, y = residuals,
                                  text = paste("State:", states, "<br>Predicted:", round(predicted_values, 2), "Cr<br>Residual:", round(residuals, 2), "Cr"))) +
      geom_point(alpha = 0.7, color = "darkgreen") +
      geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 1) +
      labs(title = paste("Residuals vs. Predicted for", names(input$plot_model_select)[input$plot_model_select == selected_pred_col]),
           x = "Predicted Tax Collection (₹ Crore)",
           y = "Residuals (This Study's Actual - Predicted)") +
      theme_minimal() +
      theme(text = element_text(family = "sans", size = 14),
            plot.title = element_text(hjust = 0.5, face = "bold"))
    
    ggplotly(p, tooltip = "text")
  })
  
  # Static State-Specific Recommendations (Pre-defined Text Block with dynamic values)
  output$state_recommendations <- renderUI({
    req(model_results(), input$selected_state)
    results_data <- model_results()$test_data_with_preds
    metrics <- model_results()$metrics_table
    rf_model_obj <- model_results()$rf_model # Get the raw randomForest object
    xgb_model_obj <- model_results()$xgb_model # Get the raw xgboost object
    
    selected_state_data <- results_data %>% filter(states == input$selected_state)
    
    if (nrow(selected_state_data) == 0) {
      return(HTML("<p>No data available for the selected state in the test set.</p>"))
    }
    
    actual_tax <- selected_state_data$tcny
    best_model_name <- rownames(metrics)[which.min(metrics[, "RMSE"])]
    best_model_pred_col <- switch(best_model_name,
                                  "RandomForest" = "rf_pred",
                                  "XGBoost" = "xgb_pred",
                                  "LASSO" = "lasso_pred",
                                  "GMM" = "gmm_predicted",
                                  "rf_pred")
    
    predicted_tax <- selected_state_data[[best_model_pred_col]]
    residual <- actual_tax - predicted_tax
    
    # Get top feature from Random Forest (or XGBoost if RF not available/important)
    top_feature_rf <- NULL
    if (!is.null(rf_model_obj) && "randomForest" %in% class(rf_model_obj)) {
      if (!is.null(rf_model_obj$importance)) {
        rf_imp_df <- as.data.frame(rf_model_obj$importance)
        if (nrow(rf_imp_df) > 0) {
          top_feature_rf <- rownames(rf_imp_df)[which.max(rf_imp_df$`%IncMSE`)]
        }
      }
    }
    
    top_feature_xgb <- NULL
    if (!is.null(xgb_model_obj) && "xgb.Booster" %in% class(xgb_model_obj)) {
      xgb_imp <- xgb.importance(model = xgb_model_obj)
      if (nrow(xgb_imp) > 0) {
        top_feature_xgb <- xgb_imp$Feature[1]
      }
    }
    
    main_focus_feature <- if (!is.null(top_feature_rf)) top_feature_rf else top_feature_xgb
    if (is.null(main_focus_feature)) {
      main_focus_feature_text <- "key economic indicators influencing tax revenue"
    } else {
      feature_map <- c(
        "log_pcepf" = "Per Capita EPF subscribers",
        "log_upiu" = "Total UPI users",
        "log_service" = "Share of the Service Sector in Gross Value Added",
        "log_pcbd" = "Per Capita Bank Deposits",
        "log_lrrurr" = "Interaction of Literacy Rate and Urbanization Rate",
        "tax_lag" = "Previous Year's Tax Collection"
      )
      descriptive_feature_name <- feature_map[main_focus_feature]
      if (is.na(descriptive_feature_name)) {
        main_focus_feature_text <- paste0("the feature '", main_focus_feature, "'")
      } else {
        main_focus_feature_text <- descriptive_feature_name
      }
    }
    
    recommendation_text <- ""
    if (is.na(residual)) {
      recommendation_text <- "<p>Cannot provide specific recommendations due to missing data for this state.</p>"
    } else if (abs(residual) < 0.05 * actual_tax) {
      recommendation_text <- paste0(
        "<h4>Insights for ", input$selected_state, " (by this Study):</h4>",
        "<p>For <b>", input$selected_state, "</b>, this study suggests that the actual tax collection (₹", round(actual_tax, 2), " Crore) is remarkably close to the predicted value (₹", round(predicted_tax, 2), " Crore) by the <b>", best_model_name, "</b> model. This indicates a stable and predictable tax revenue stream, suggesting that current economic trends and policy frameworks are largely aligned with expectations.</p>",
        "<p><b>Policy Implication:</b> While no immediate alarms are raised, continuous monitoring of ", main_focus_feature_text, " and other economic indicators is vital. Policymakers should focus on sustaining this stability, proactively identifying emerging economic shifts, and planning for long-term growth strategies to ensure future tax revenue remains robust and predictable. This state can serve as a benchmark for effective fiscal management."
      )
    } else if (residual > 0) {
      recommendation_text <- paste0(
        "<h4>Insights for ", input$selected_state, " (by this Study):</h4>",
        "<p>For <b>", input$selected_state, "</b>, this study suggests that the actual tax collection (₹", round(actual_tax, 2), " Crore) significantly exceeded the predicted value (₹", round(predicted_tax, 2), " Crore) by the <b>", best_model_name, "</b> model. This indicates the state <b>overperformed</b> relative to expectations, showing a positive deviation of ₹", round(residual, 2), " Crore.</p>",
        "<p><b>Policy Implication:</b> This positive deviation suggests strong underlying economic activity or successful policy interventions. It's crucial for policymakers in ", input$selected_state, " to conduct a detailed analysis to identify the specific drivers behind this exceptional performance. This could involve:<ul>",
        "<li>Investigating recent fiscal policies, tax reforms, or economic stimulus packages that may have yielded unexpected benefits.</li>",
        "<li>Analyzing specific sectoral booms (e.g., in digital services, manufacturing, tourism, or agriculture) that significantly contributed to revenue growth.</li>",
        "<li>Understanding how factors like ", main_focus_feature_text, " performed exceptionally well in this state.</li></ul>",
        "Documenting these successes and understanding their causal mechanisms can provide invaluable lessons for future policy formulation within the state and potentially serve as a best practice model for other regions seeking to enhance their tax base."
      )
    } else {
      recommendation_text <- paste0(
        "<h4>Insights for ", input$selected_state, " (by this Study):</h4>",
        "<p>For <b>", input$selected_state, "</b>, this study suggests that the actual tax collection (₹", round(actual_tax, 2), " Crore) fell short of the predicted value (₹", round(predicted_tax, 2), " Crore) by the <b>", best_model_name, "</b> model. This indicates the state <b>underperformed</b> relative to expectations, showing a negative deviation of ₹", round(residual, 2), " Crore.</p>",
        "<p><b>Policy Implication:</b> This underperformance signals potential challenges or inefficiencies in revenue generation that require urgent attention. Policymakers should focus on understanding the root causes of this shortfall. Key areas for investigation and potential intervention include:<ul>",
        "<li><b>Tax Administration & Compliance:</b> Reviewing the efficiency of tax collection mechanisms, identifying bottlenecks, and exploring initiatives to improve compliance rates.</li>",
        "<li><b>Economic Sector Analysis:</b> Pinpointing any slowdowns, contractions, or structural issues in key economic sectors that are primary contributors to tax revenue. If ", main_focus_feature_text, " is a highly influential factor, a deeper analysis into its trends and policy implications within ", input$selected_state, " is warranted.</li>",
        "<li><b>Policy Effectiveness:</b> Evaluating the impact of recent state-level economic or fiscal policies to identify any unintended negative consequences or areas where policies did not achieve their intended revenue-generating effects.</li>",
        "<li><b>Investment & Growth Stimuli:</b> Considering new initiatives to stimulate economic activity, attract investment, or support industries that can broaden and strengthen the tax base.</li></ul>",
        "Targeted interventions based on this analysis are crucial to bring tax collection back in line with economic potential."
      )
    }
    
    HTML(recommendation_text)
  })
  
  # NEW: Feature Distribution Plot (Interactive with Plotly)
  output$feature_distribution_plot <- renderPlotly({
    req(raw_data(), input$dist_feature_select)
    plot_data <- raw_data()
    selected_feature <- input$dist_feature_select
    
    if (!selected_feature %in% colnames(plot_data) || all(is.na(plot_data[[selected_feature]]))) {
      return(NULL)
    }
    
    feature_display_names <- c(
      "tcny" = "Tax Collection (₹ Crore)",
      "pcEPF" = "Per Capita EPF Subscribers",
      "UPIu" = "Total UPI Users",
      "serviceGSVAr" = "Service Sector Gross Value Added",
      "pcbd" = "Per Capita Bank Deposits",
      "lrr" = "Literacy Rate (LRR)",
      "urr" = "Urbanization Rate (URR)"
    )
    
    plot_title <- paste("Distribution of", feature_display_names[selected_feature])
    x_axis_label <- feature_display_names[selected_feature]
    
    p <- ggplot(plot_data, aes(x = !!sym(selected_feature))) +
      geom_histogram(aes(y = after_stat(density), text = !!sym(selected_feature)),
                     binwidth = diff(range(plot_data[[selected_feature]], na.rm = TRUE))/30, fill = "lightblue", color = "black", alpha = 0.7) +
      geom_density(color = "darkblue", size = 1) +
      labs(title = plot_title,
           x = x_axis_label,
           y = "Density") +
      theme_minimal() +
      theme(text = element_text(family = "sans", size = 14),
            plot.title = element_text(hjust = 0.5, face = "bold"))
    
    ggplotly(p, tooltip = "text") %>%
      layout(hoverlabel = list(namelength = -1))
  })
  
  # NEW: Feature Distribution Interpretation
  output$feature_distribution_interpretation <- renderUI({
    req(raw_data(), input$dist_feature_select)
    selected_feature <- input$dist_feature_select
    feature_display_name <- switch(selected_feature,
                                   "tcny" = "Tax Collection (TCNY)",
                                   "pcEPF" = "Per Capita EPF Subscribers",
                                   "UPIu" = "Total UPI Users",
                                   "serviceGSVAr" = "Service Sector Gross Value Added",
                                   "pcbd" = "Per Capita Bank Deposits",
                                   "lrr" = "Literacy Rate (LRR)",
                                   "urr" = "Urbanization Rate (URR)",
                                   selected_feature)
    
    data_values <- na.omit(raw_data()[[selected_feature]])
    
    if (length(data_values) == 0) {
      return(HTML(paste0("<p>No data available for '", feature_display_name, "' to analyze its distribution.</p>")))
    }
    
    mean_val <- mean(data_values)
    median_val <- median(data_values)
    sd_val <- sd(data_values)
    min_val <- min(data_values)
    max_val <- max(data_values)
    
    skewness_val <- e1071::skewness(data_values)
    
    distribution_insight <- ""
    if (abs(skewness_val) < 0.5) {
      distribution_insight <- "The data appears fairly symmetrical, meaning values are evenly distributed around the average."
    } else if (skewness_val > 0.5) {
      distribution_insight <- "The data is skewed to the right (positive skew), indicating a longer tail on the right side. This means there are a few unusually high values that pull the average up."
    } else {
      distribution_insight <- "The data is skewed to the left (negative skew), indicating a longer tail on the left side. This means there are a few unusually low values that pull the average down."
    }
    
    HTML(paste0(
      "<h4>Understanding the Distribution of ", feature_display_name, ":</h4>",
      "<p>This plot shows how values for '", feature_display_name, "' are spread across the dataset. The taller bars indicate more frequent values, while the overall shape tells us about its typical range and any unusual concentrations.</p>",
      "<ul>",
      "<li><b>Average Value:</b> Approximately ₹", round(mean_val, 2), " Crore.</li>",
      "<li><b>Typical Range (Standard Deviation):</b> Values typically vary by about ₹", round(sd_val, 2), " Crore from the average.</li>",
      "<li><b>Minimum Value:</b> ₹", round(min_val, 2), " Crore.</li>",
      "<li><b>Maximum Value:</b> ₹", round(max_val, 2), " Crore.</li>",
      "<li><b>Shape of Distribution:</b> ", distribution_insight, " Understanding this helps us see if most states fall within a similar range or if there are significant differences.</li>",
      "</ul>"
    ))
  })
  
  # NEW: Actual vs. Predicted Tax Collection by State (Interactive with Plotly)
  output$actual_vs_predicted_state_plot <- renderPlotly({
    req(model_results(), input$state_comparison_model)
    plot_data <- model_results()$test_data_with_preds %>%
      select(states, tcny, predicted = input$state_comparison_model) %>%
      mutate(states = factor(states, levels = states[order(tcny)]))
    
    model_display_name <- names(input$state_comparison_model)[input$state_comparison_model == input$state_comparison_model]
    
    p <- ggplot(plot_data, aes(x = states)) +
      geom_bar(aes(y = tcny, fill = "Actual Tax",
                   text = paste("State:", states, "<br>Actual Tax:", round(tcny, 2), "Cr")),
               stat = "identity", position = "dodge", alpha = 0.8) +
      geom_bar(aes(y = predicted, fill = paste(model_display_name, "Predicted"),
                   text = paste("State:", states, "<br>Predicted Tax:", round(predicted, 2), "Cr")),
               stat = "identity", position = "dodge", alpha = 0.8) +
      labs(title = paste("Actual vs.", model_display_name, "Predicted Tax Collection by State (2023)"),
           x = "State",
           y = "Tax Collection (₹ Crore)",
           fill = "Type") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
            text = element_text(family = "sans", size = 14),
            plot.title = element_text(hjust = 0.5, face = "bold"),
            legend.position = "bottom") +
      scale_fill_manual(values = setNames(c("#1f77b4", "#ff7f0e"), c("Actual Tax", paste(model_display_name, "Predicted"))))
    
    ggplotly(p, tooltip = "text") %>%
      layout(hovermode = "closest")
  })
  
  # NEW: Actual vs. Predicted State Interpretation
  output$actual_vs_predicted_state_interpretation <- renderUI({
    req(model_results(), input$state_comparison_model)
    model_display_name <- names(input$state_comparison_model)[input$state_comparison_model == input$state_comparison_model]
    HTML(paste0(
      "<h4>State-Specific Performance Overview:</h4>",
      "<p>This bar chart directly compares the <b>Actual Tax Collection</b> for each state in 2023 against the tax collection predicted by the <b>", model_display_name, "</b> model. Each pair of bars allows you to quickly see how well the model performed for individual states.</p>",
      "<ul>",
      "<li><b>Close Match:</b> If the 'Actual Tax' bar (blue) and the '", model_display_name, " Predicted' bar (orange) are nearly the same height, it means the model made a very accurate forecast for that state.</li>",
      "<li><b>Underprediction (Actual > Predicted):</b> If the 'Actual Tax' bar is significantly taller than the '", model_display_name, " Predicted' bar, the model underestimated the tax collection for that state. This could indicate a stronger-than-expected economic performance or successful local policy initiatives that the model didn't fully capture.</li>",
      "<li><b>Overprediction (Actual < Predicted):</b> If the 'Actual Tax' bar is noticeably shorter than the '", model_display_name, " Predicted' bar, the model overestimated the tax collection. This might suggest an unexpected economic slowdown, challenges in tax administration, or other factors that led to lower-than-expected revenue.</li>",
      "</ul>",
      "<p>Hover over each bar to see the exact tax collection values for a precise comparison.</p>"
    ))
  })
  
  
  # NEW: Residuals by State Plot (Interactive with Plotly)
  output$residuals_by_state_plot <- renderPlotly({
    req(model_results(), input$state_comparison_model)
    plot_data <- model_results()$test_data_with_preds %>%
      select(states, tcny, predicted = input$state_comparison_model) %>%
      mutate(residual = tcny - predicted) %>%
      mutate(states = factor(states, levels = states[order(residual)]))
    
    model_display_name <- names(input$state_comparison_model)[input$state_comparison_model == input$state_comparison_model]
    
    p <- ggplot(plot_data, aes(x = states, y = residual, fill = residual > 0,
                               text = paste("State:", states, "<br>Residual:", round(residual, 2), "Cr"))) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 1) +
      labs(title = paste("Prediction Errors (Residuals) by State for", model_display_name, "(2023)"),
           x = "State",
           y = "Residual (Actual - Predicted) (₹ Crore)",
           fill = "Underpredicted (Actual > Predicted)") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
            text = element_text(family = "sans", size = 14),
            plot.title = element_text(hjust = 0.5, face = "bold"),
            legend.position = "bottom") +
      scale_fill_manual(values = c("TRUE" = "#2ca02c", "FALSE" = "#d62728"),
                        labels = c("TRUE" = "Underpredicted (Actual > Predicted)", "FALSE" = "Overpredicted (Actual < Predicted)"))
    
    ggplotly(p, tooltip = "text") %>%
      layout(hovermode = "closest")
  })
  
  # NEW: Residuals by State Interpretation
  output$residuals_by_state_interpretation <- renderUI({
    req(model_results(), input$state_comparison_model)
    model_display_name <- names(input$state_comparison_model)[input$state_comparison_model == input$state_comparison_model]
    HTML(paste0(
      "<h4>Understanding Prediction Errors by State:</h4>",
      "<p>This bar chart shows the <b>residuals</b>, which are the differences between the Actual Tax Collection and the tax predicted by the <b>", model_display_name, "</b> model for each state in 2023. This helps us pinpoint exactly where the model's forecasts were most accurate or most off the mark.</p>",
      "<ul>",
      "<li><b>Positive Bars (Green):</b> A bar extending upwards means the model <b>underestimated</b> the tax collection (Actual Tax > Predicted Tax). The taller the green bar, the more the model missed the actual revenue on the low side.</li>",
      "<li><b>Negative Bars (Red):</b> A bar extending downwards means the model <b>overestimated</b> the tax collection (Actual Tax < Predicted Tax). The longer the red bar, the more the model predicted too much revenue.</li>",
      "<li><b>Bars Near Zero:</b> States with bars very close to the horizontal line (zero) indicate that the model's prediction was highly accurate for those states.</li>",
      "</ul>",
      "<p>Analyzing these residuals helps this study identify specific states where the model needs further refinement or where unique, uncaptured factors might be at play. Hover over each bar to see the exact residual value.</p>"
    ))
  })
  
  # NEW: Reactive for state-wise prediction accuracy analysis
  state_accuracy_analysis <- reactive({
    req(model_results())
    results_data <- model_results()$test_data_with_preds
    metrics <- model_results()$metrics_table
    
    best_model_name <- rownames(metrics)[which.min(metrics[, "RMSE"])]
    best_model_pred_col <- switch(best_model_name,
                                  "RandomForest" = "rf_pred",
                                  "XGBoost" = "xgb_pred",
                                  "LASSO" = "lasso_pred",
                                  "GMM" = "gmm_predicted",
                                  "rf_pred")
    
    analysis_data <- results_data %>%
      mutate(
        predicted_by_best_model = .data[[best_model_pred_col]],
        absolute_error = abs(tcny - predicted_by_best_model),
        ape = ifelse(tcny != 0, (absolute_error / tcny) * 100, NA)
      )
    
    analysis_data <- analysis_data %>%
      mutate(
        accuracy_category = case_when(
          ape <= 5 ~ "Near Perfect (APE <= 5%)",
          ape > 5 & ape <= 15 ~ "Somewhat Close (5% < APE <= 15%)",
          ape > 15 ~ "Significant Deviation (APE > 15%)",
          is.na(ape) ~ "N/A (Due to Zero Actual Tax or Missing Prediction)"
        )
      ) %>%
      select(states, year, tcny, predicted_by_best_model, absolute_error, ape, accuracy_category,
             pcEPF, UPIu, serviceGSVAr, pcbd, lrr, urr, tax_lag,
             log_pcepf, log_upiu, log_service, log_pcbd, log_lrrurr)
    
    category_summary <- analysis_data %>%
      group_by(accuracy_category) %>%
      summarise(
        Num_States = n(),
        Avg_Actual_Tax = mean(tcny, na.rm = TRUE),
        Avg_APE = mean(ape, na.rm = TRUE),
        Avg_pcEPF = mean(pcEPF, na.rm = TRUE),
        Avg_UPIu = mean(UPIu, na.rm = TRUE),
        Avg_serviceGSVAr = mean(serviceGSVAr, na.rm = TRUE),
        Avg_pcbd = mean(pcbd, na.rm = TRUE),
        Avg_lrr = mean(lrr, na.rm = TRUE),
        Avg_urr = mean(urr, na.rm = TRUE),
        Avg_tax_lag = mean(tax_lag, na.rm = TRUE),
        States_List = paste(sort(unique(states)), collapse = ", ")
      ) %>%
      arrange(factor(accuracy_category, levels = c("Near Perfect (APE <= 5%)", "Somewhat Close (5% < APE <= 15%)", "Significant Deviation (APE > 15%)", "N/A (Due to Zero Actual Tax or Missing Prediction)")))
    
    list(
      detailed_state_data = analysis_data,
      category_summary = category_summary,
      best_model_used = best_model_name
    )
  })
  
  # NEW: Render the detailed state accuracy table
  output$state_accuracy_table <- renderDT({
    req(state_accuracy_analysis())
    datatable(state_accuracy_analysis()$detailed_state_data %>%
                select(states, year, `Actual Tax (Cr)` = tcny, `Predicted Tax (Cr)` = predicted_by_best_model, `APE (%)` = ape, `Accuracy Category` = accuracy_category) %>%
                mutate(`APE (%)` = round(`APE (%)`, 2)),
              options = list(pageLength = 10, scrollX = TRUE),
              caption = paste("State-wise Prediction Accuracy (using", state_accuracy_analysis()$best_model_used, "Model)"))
  })
  
  # NEW: Render the pattern interpretation for state accuracy
  output$accuracy_pattern_interpretation <- renderUI({
    req(state_accuracy_analysis())
    summary_data <- state_accuracy_analysis()$category_summary
    best_model <- state_accuracy_analysis()$best_model_used
    
    html_output <- paste0("<h4>Patterns in Model Accuracy by State (Using ", best_model, " Model):</h4>")
    html_output <- paste0(html_output, "<p>This analysis categorizes states based on how accurately the <b>", best_model, "</b> model predicted their tax collection in 2023. By examining the average characteristics of states within each category, we can gain insights into what factors might lead to more predictable or less predictable tax revenues.</p>")
    
    feature_descriptions <- list(
      "pcEPF" = "Per Capita EPF subscribers, representing formal employment and social security coverage.",
      "UPIu" = "Total UPI users, reflecting digital financial penetration and economic formalization, which reduces tax evasion and boosts compliance.",
      "serviceGSVAr" = "Share of the Service Sector in Gross Value Added (GSVA) at constant prices, capturing the presence of high-income professionals and formal enterprises within direct taxation.",
      "pcbd" = "Per Capita Bank Deposits, proxying financial inclusion and saving behavior, linked to higher taxable income.",
      "lrr" = "Literacy Rate (LRR), representing combined human development and economic modernity.",
      "urr" = "Urbanization Rate (URR), representing combined human development and economic modernity.",
      "tax_lag" = "Previous Year's Tax Collection, reflecting historical revenue trends."
    )
    interaction_description <- "Interaction of Literacy Rate and Urbanization Rate, representing combined human development and economic modernity, hypothesized to enhance administrative capacity, civic awareness, and tax compliance."
    
    
    near_perfect_summary <- summary_data %>% filter(accuracy_category == "Near Perfect (APE <= 5%)")
    somewhat_close_summary <- summary_data %>% filter(accuracy_category == "Somewhat Close (5% < APE <= 15%)")
    significant_deviation_summary <- summary_data %>% filter(accuracy_category == "Significant Deviation (APE > 15%)")
    
    if (nrow(near_perfect_summary) > 0) {
      html_output <- paste0(html_output, "<h5>States with Near Perfect Forecasts (APE <= 5%)</h5>")
      html_output <- paste0(html_output, "<p>These states had tax collections closely matched by our model, indicating highly predictable revenue streams. This suggests a stable economic environment where the key drivers of tax collection are well-captured by the model.</p>")
      html_output <- paste0(html_output, "<p><b>States:</b> ", near_perfect_summary$States_List, "</p>")
      html_output <- paste0(html_output, "<h6>Average Characteristics:</h6><ul>")
      for (feature_col in c("pcEPF", "UPIu", "serviceGSVAr", "pcbd", "lrr", "urr", "tax_lag")) {
        avg_val <- round(near_perfect_summary[[paste0("Avg_", feature_col)]], 2)
        html_output <- paste0(html_output, "<li><b>", names(feature_descriptions[feature_col]), ":</b> ", avg_val, "</li>")
      }
      if ("log_lrrurr" %in% colnames(near_perfect_summary)) {
        html_output <- paste0(html_output, "<li><b>Literacy Rate * Urbanization Rate (Interaction Term):</b> This combined factor also contributes to the predictability.</li>")
      }
      html_output <- paste0(html_output, "</ul>")
    }
    
    if (nrow(somewhat_close_summary) > 0) {
      html_output <- paste0(html_output, "<h5>States with Somewhat Close Forecasts (5% < APE <= 15%)</h5>")
      html_output <- paste0(html_output, "<p>For these states, the model provided reasonably accurate forecasts, but with some room for improvement. Small, uncaptured economic shifts or minor policy impacts might contribute to these deviations.</p>")
      html_output <- paste0(html_output, "<p><b>States:</b> ", somewhat_close_summary$States_List, "</p>")
      html_output <- paste0(html_output, "<h6>Average Characteristics:</h6><ul>")
      for (feature_col in c("pcEPF", "UPIu", "serviceGSVAr", "pcbd", "lrr", "urr", "tax_lag")) {
        avg_val <- round(somewhat_close_summary[[paste0("Avg_", feature_col)]], 2)
        html_output <- paste0(html_output, "<li><b>", names(feature_descriptions[feature_col]), ":</b> ", avg_val, "</li>")
      }
      if ("log_lrrurr" %in% colnames(somewhat_close_summary)) {
        html_output <- paste0(html_output, "<li><b>Literacy Rate * Urbanization Rate (Interaction Term):</b> This combined factor also contributes.</li>")
      }
      html_output <- paste0(html_output, "</ul>")
    }
    
    if (nrow(significant_deviation_summary) > 0) {
      html_output <- paste0(html_output, "<h5>States with Significant Deviations (APE > 15%)</h5>")
      html_output <- paste0(html_output, "<p>These states showed the largest differences between actual and predicted tax collections. This indicates that the model struggled to accurately forecast their revenue, likely due to unique economic events, uncaptured policy changes, or inherent volatility in their tax base.</p>")
      html_output <- paste0(html_output, "<p><b>States:</b> ", significant_deviation_summary$States_List, "</p>")
      html_output <- paste0(html_output, "<h6>Average Characteristics:</h6><ul>")
      for (feature_col in c("pcEPF", "UPIu", "serviceGSVAr", "pcbd", "lrr", "urr", "tax_lag")) {
        avg_val <- round(significant_deviation_summary[[paste0("Avg_", feature_col)]], 2)
        html_output <- paste0(html_output, "<li><b>", names(feature_descriptions[feature_col]), ":</b> ", avg_val, "</li>")
      }
      if ("log_lrrurr" %in% colnames(significant_deviation_summary)) {
        html_output <- paste0(html_output, "<li><b>Literacy Rate * Urbanization Rate (Interaction Term):</b> This combined factor is also present.</li>")
      }
      html_output <- paste0(html_output, "</ul>")
    }
    
    if (nrow(near_perfect_summary) > 0 && nrow(significant_deviation_summary) > 0) {
      html_output <- paste0(html_output,
                            "<p><b>Key Patterns & Policy Implications:</b></p>",
                            "<p>Comparing the characteristics of these categories provides valuable insights for policymakers (this study):</p>",
                            "<ul>")
      
      compare_features <- c("pcEPF", "UPIu", "serviceGSVAr", "pcbd", "lrr", "urr", "tax_lag")
      for (f in compare_features) {
        avg_np <- near_perfect_summary[[paste0("Avg_", f)]]
        avg_sd <- significant_deviation_summary[[paste0("Avg_", f)]]
        
        if (!is.na(avg_np) && !is.na(avg_sd)) {
          if (avg_np > avg_sd * 1.1) {
            html_output <- paste0(html_output, "<li>States with generally <b>higher ", names(feature_descriptions[f]), "</b> tend to have more predictable tax revenues. This suggests that robust and formalized economic activity (e.g., more formal employment, higher digital adoption) leads to more stable and forecastable tax bases.</li>")
          } else if (avg_np < avg_sd * 0.9) {
            html_output <- paste0(html_output, "<li>States with generally <b>lower ", names(feature_descriptions[f]), "</b> might be associated with more predictable tax revenues, possibly due to simpler or less volatile economic structures.</li>")
          } else {
            html_output <- paste0(html_output, "<li>Differences in <b>", names(feature_descriptions[f]), "</b> between predictable and less predictable states are not consistently large enough to draw strong conclusions from this analysis alone.</li>")
          }
        }
      }
      if ("log_lrrurr" %in% colnames(near_perfect_summary) && "log_lrrurr" %in% colnames(significant_deviation_summary)) {
        avg_lrrurr_np <- near_perfect_summary[["Avg_log_lrrurr"]]
        avg_lrrurr_sd <- significant_deviation_summary[["Avg_log_lrrurr"]]
        if (!is.na(avg_lrrurr_np) && !is.na(avg_lrrurr_sd)) {
          if (avg_lrrurr_np > avg_lrrurr_sd * 1.05) {
            html_output <- paste0(html_output, "<li>States with higher average <b>Literacy Rate * Urbanization Rate (Interaction)</b> tend to be more predictable. This aligns with the hypothesis that combined human development and economic modernity enhance administrative capacity and tax compliance.</li>")
          } else if (avg_lrrurr_np < avg_lrrurr_sd * 0.95) {
            html_output <- paste0(html_output, "<li>States with lower average <b>Literacy Rate * Urbanization Rate (Interaction)</b> might be more predictable, suggesting that complex socio-economic dynamics can sometimes introduce forecasting challenges.</li>")
          }
        }
      }
      
      
      html_output <- paste0(html_output,
                            "</ul>",
                            "<p><b>Next Steps for Policymakers:</b> For states with 'Significant Deviations', it's crucial to investigate unique local factors not captured by these general economic indicators. This could involve deep-dives into specific policy changes, unforeseen economic shocks, or structural issues within their tax administration. Understanding these nuances will be key to improving future tax forecasting accuracy and designing targeted interventions.</p>")
    } else {
      html_output <- paste0(html_output, "<p>Insufficient data in 'Near Perfect' or 'Significant Deviation' categories to provide comparative insights. Please ensure both categories have states for a meaningful comparison.</p>")
    }
    
    HTML(html_output)
  })
  
}

# Run the application
shinyApp(ui = ui, server = server)
