# ONE-TIME SETUP SCRIPT: Run this once to pre-train and save models

# ==========================================
# 📦 Load Libraries (ensure these are installed)
# ==========================================
if (!requireNamespace("readxl", quietly = TRUE)) install.packages("readxl")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("randomForest", quietly = TRUE)) install.packages("randomForest")
if (!requireNamespace("xgboost", quietly = TRUE)) install.packages("xgboost")
if (!requireNamespace("glmnet", quietly = TRUE)) install.packages("glmnet")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret") # Still useful for varImp if models were caret-trained
if (!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071") # For skewness

library(readxl)
library(dplyr)
library(randomForest)
library(xgboost)
library(glmnet)
library(caret) # Needed for varImp for randomForest and xgboost objects
library(e1071) # For skewness

# Define the directory to save models
# This assumes 'models' folder will be created in the same directory as this script.
model_dir <- "models"
if (!dir.exists(model_dir)) {
  dir.create(model_dir)
  message(paste("Created directory:", model_dir))
}

# ==========================================
# 📂 Load Data
# ==========================================
file_path <- "ForecastData.xlsx"
if (!file.exists(file_path)) {
  stop(paste("Error: 'ForecastData.xlsx' not found in the current directory (", getwd(), "). Please place it here."))
}
data <- read_excel(file_path)

# ==========================================
# 🪜 Data Transformation
# ==========================================
data <- data %>%
  mutate(
    log_pcepf    = log(pcEPF),
    log_upiu     = log(UPIu),
    log_service  = log(serviceGSVAr),
    log_pcbd     = log(pcbd),
    log_lrrurr   = log(lrr * urr),
    tax_lag      = lag(tcny) # TCNY(-1)
  ) %>%
  filter(!is.na(tax_lag))

# ==========================================
# 🪜 Split into Training (2018–2022) and Test (2023)
# ==========================================
train <- data %>% filter(year < 2023)
test  <- data %>% filter(year == 2023)

features <- c("log_pcepf", "log_upiu", "log_service", "log_pcbd", "log_lrrurr", "tax_lag")
target <- "tcny"

# Validate columns
if (!all(features %in% colnames(train)) || !target %in% colnames(train) ||
    !all(features %in% colnames(test)) || !target %in% colnames(test) ||
    !"gmm_predicted" %in% colnames(test)) {
  stop("Missing required columns in data after splitting. Please check your 'ForecastData.xlsx'.")
}

# ==========================================
# 🌲 Random Forest Model (Full Training)
# ==========================================
message("Training Random Forest model...")
set.seed(42)
train_rf_data <- train[, c(features, target)]
train_rf_data <- na.omit(train_rf_data) # Remove NAs for training
rf_model <- randomForest(tcny ~ ., data = train_rf_data, ntree = 500, importance = TRUE)
test$rf_pred <- predict(rf_model, newdata = test[, features])

# Calculate prediction intervals for Random Forest
rf_all_preds <- predict(rf_model, newdata = test[, features], predict.all = TRUE)$individual
test$rf_lower_90 <- apply(rf_all_preds, 1, function(x) quantile(x, 0.05, na.rm = TRUE))
test$rf_upper_90 <- apply(rf_all_preds, 1, function(x) quantile(x, 0.95, na.rm = TRUE))
test$rf_lower_95 <- apply(rf_all_preds, 1, function(x) quantile(x, 0.025, na.rm = TRUE))
test$rf_upper_95 <- apply(rf_all_preds, 1, function(x) quantile(x, 0.975, na.rm = TRUE))
message("Random Forest training complete.")

# ==========================================
# 🚀 XGBoost Model (Full Training)
# ==========================================
message("Training XGBoost model...")
train_xgb_data <- train[, c(features, target)]
train_xgb_data <- na.omit(train_xgb_data)

if (nrow(train_xgb_data) == 0) {
  warning("No complete cases for XGBoost training after NA removal. XGBoost model will not be trained.")
  test$xgb_pred <- NA
  xgb_model <- NULL
} else {
  train_matrix <- xgb.DMatrix(data = as.matrix(train_xgb_data[, features]), label = train_xgb_data[[target]])
  xgb_model <- xgboost(data = train_matrix, nrounds = 100, objective = "reg:squarederror", verbose = 0)
  
  test_features_matrix <- as.matrix(test[, features])
  test_features_matrix[!is.finite(test_features_matrix)] <- NA
  test_matrix <- xgb.DMatrix(data = test_features_matrix)
  test$xgb_pred <- predict(xgb_model, test_matrix)
  message("XGBoost training complete.")
}

# ==========================================
# 📐 LASSO Regression (Full Training)
# ==========================================
message("Training LASSO Regression model...")
X_train <- as.matrix(train[, features])
y_train <- train[[target]]
na_rows_train <- unique(c(which(is.na(X_train), arr.ind = TRUE)[,1], which(is.na(y_train))))
if (length(na_rows_train) > 0) {
  X_train <- X_train[-na_rows_train,]
  y_train <- y_train[-na_rows_train]
}
X_test  <- as.matrix(test[, features])
if (any(is.na(X_test))) {
  print("Warning: NAs found in LASSO test features. Predictions might be affected.")
}

lasso_model <- cv.glmnet(X_train, y_train, alpha = 1)

# Predict on 2023
test$lasso_pred <- as.vector(predict(lasso_model, s = "lambda.min", newx = X_test))
message("LASSO Regression training complete.")

# ==========================================
# 📊 Calculate and Save Metrics
# ==========================================
message("Calculating and saving metrics...")
get_metrics <- function(actual, predicted) {
  valid_indices <- !is.na(actual) & !is.na(predicted) & !is.infinite(actual) & !is.infinite(predicted)
  if (sum(valid_indices) == 0) return(c(RMSE = NA, MAE = NA, MAPE = NA))
  actual_valid <- actual[valid_indices]
  predicted_valid <- predicted[valid_indices]
  rmse <- sqrt(mean((actual_valid - predicted_valid)^2))
  mae  <- mean(abs(actual_valid - predicted_valid))
  mape <- mean(abs((actual_valid - predicted_valid) / actual_valid[actual_valid != 0])) * 100
  if (is.nan(mape) || is.infinite(mape)) mape <- NA
  return(c(RMSE = rmse, MAE = mae, MAPE = mape))
}

metrics_gmm   <- tryCatch(get_metrics(test$tcny, test$gmm_predicted), error = function(e) { message(paste("Error calculating GMM metrics:", e$message)); c(RMSE = NA, MAE = NA, MAPE = NA) })
metrics_rf    <- tryCatch(get_metrics(test$tcny, test$rf_pred), error = function(e) { message(paste("Error calculating RF metrics:", e$message)); c(RMSE = NA, MAE = NA, MAPE = NA) })
metrics_xgb   <- tryCatch(get_metrics(test$tcny, test$xgb_pred), error = function(e) { message(paste("Error calculating XGB metrics:", e$message)); c(RMSE = NA, MAE = NA, MAPE = NA) })
metrics_lasso <- tryCatch(get_metrics(test$tcny, test$lasso_pred), error = function(e) { message(paste("Error calculating LASSO metrics:", e$message)); c(RMSE = NA, MAE = NA, MAPE = NA) })

metrics_table <- rbind(
  GMM          = metrics_gmm,
  RandomForest = metrics_rf,
  XGBoost      = metrics_xgb,
  LASSO        = metrics_lasso
)
write.csv(as.data.frame(metrics_table), file.path(model_dir, "model_accuracy_metrics.csv"), row.names = TRUE)
message("Metrics saved to models/model_accuracy_metrics.csv")

# ==========================================
# 💾 Save Trained Models and Test Data with Predictions
# ==========================================
message("Saving trained models and test data...")
saveRDS(rf_model, file.path(model_dir, "rf_model.rds"))
saveRDS(xgb_model, file.path(model_dir, "xgb_model.rds"))
saveRDS(lasso_model, file.path(model_dir, "lasso_model.rds"))
saveRDS(test, file.path(model_dir, "test_data_with_preds.rds"))
message("Models and test data saved successfully in the 'models' directory.")

message("\nOne-time setup complete. You can now run the Shiny app (app.R) for faster loading.")

