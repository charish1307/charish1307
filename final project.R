# Load necessary libraries
library(caret)
library(randomForest)
library(xgboost)
library(dplyr)
library(corrplot)
library(Amelia)
library(ggplot2)
library(GGally)
library(ROCR)
library(SHAPforxgboost)
library(doParallel) 
library(pROC)
library(tidyr)
# Load the new dataset
new_diabetes_data <- read.csv("diabetes.csv")
# Check the structure 
str(new_diabetes_data)
#summary of the dataset
summary(new_diabetes_data)
missmap(new_diabetes_data, main = "Missing Data Heatmap", col = c("red", "grey"))
# Class balance
ggplot(new_diabetes_data, aes(x = factor(Outcome))) +
  geom_bar(fill = c("skyblue", "tomato")) +
  labs(title = "Distribution of Outcome (Diabetes)", x = "Outcome", y = "Count") +
  scale_x_discrete(labels = c("No Diabetes", "Diabetes"))
# Histogram for all numeric variables
new_diabetes_data_long <- new_diabetes_data %>%
  pivot_longer(cols = -Outcome, names_to = "Feature", values_to = "Value")

ggplot(new_diabetes_data_long, aes(x = Value)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  facet_wrap(~ Feature, scales = "free") +
  labs(title = "Feature Distributions")
# Boxplots of features grouped by outcome
ggplot(new_diabetes_data_long, aes(x = factor(Outcome), y = Value, fill = factor(Outcome))) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free") +
  labs(title = "Boxplots of Features by Diabetes Outcome", x = "Outcome") +
  scale_fill_manual(values = c("skyblue", "tomato"))
# Pairwise relationships between features and Outcome
ggpairs(new_diabetes_data, aes(color = factor(Outcome), alpha = 0.5),
        lower = list(continuous = wrap("points", size = 0.5)))
# --- Feature Interaction Plots ---

# Example 1: Glucose vs BMI
ggplot(new_diabetes_data, aes(x = Glucose, y = BMI, color = factor(Outcome))) +
  geom_point(alpha = 0.6) +
  labs(title = "Interaction: Glucose vs BMI", color = "Outcome") +
  theme_minimal()
# Example 2: Age vs BloodPressure
ggplot(new_diabetes_data, aes(x = Age, y = BloodPressure, color = factor(Outcome))) +
  geom_point(alpha = 0.6) +
  labs(title = "Interaction: Age vs BloodPressure", color = "Outcome") +
  theme_minimal()
# --- Optional: Create Interaction Features for Modeling ---
new_diabetes_data$Glucose_BMI <- new_diabetes_data$Glucose * new_diabetes_data$BMI
new_diabetes_data$Age_BP <- new_diabetes_data$Age * new_diabetes_data$BloodPressure

# Handle missing values (if any)
preProcValues <- preProcess(new_diabetes_data, method = 'medianImpute')
new_diabetes_data <- predict(preProcValues, new_diabetes_data)

# --- Correlation Matrix ---

cor_matrix <- cor(new_diabetes_data[, -ncol(new_diabetes_data)])  # exclude Outcome column
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7, tl.col = "black", title = "Correlation Matrix of Features", mar = c(0,0,1,0))

# Feature Scaling
new_diabetes_data_scaled <- new_diabetes_data
new_diabetes_data_scaled[, -ncol(new_diabetes_data)] <- scale(new_diabetes_data[, -ncol(new_diabetes_data)])

# Check for missing values after imputation
sum(is.na(new_diabetes_data))

# Split the data into training and testing sets (80% train, 20% test)
set.seed(123)
trainIndex <- createDataPartition(new_diabetes_data$Outcome, p = 0.8, list = FALSE)
trainData <- new_diabetes_data[trainIndex, ]
testData <- new_diabetes_data[-trainIndex, ]

# Convert Outcome to factor with levels 0 and 1
trainData$Outcome <- factor(trainData$Outcome, levels = c(0, 1))
testData$Outcome <- factor(testData$Outcome, levels = c(0, 1))

# Setup cross-validation control
cv_control <- trainControl(method = "cv", number = 10)

# --- Train Logistic Regression (no parallel needed) ---
logit_model <- glm(Outcome ~ ., data = trainData, family = binomial)
logit_pred <- predict(logit_model, testData, type = "response")
logit_pred_class <- ifelse(logit_pred > 0.5, 1, 0)
logit_conf_matrix <- confusionMatrix(factor(logit_pred_class), testData$Outcome, positive = "1")
print(logit_conf_matrix)

# --- Train Random Forest (no parallel needed) ---
rf_grid <- expand.grid(mtry = c(1, 2, 3, 4, 5))
rf_model <- train(Outcome ~ ., data = trainData, method = "rf", trControl = cv_control, tuneGrid = rf_grid, importance = TRUE)
rf_pred <- predict(rf_model, testData)
rf_conf_matrix <- confusionMatrix(rf_pred, testData$Outcome, positive = "1")
print(rf_conf_matrix)

# --- Prepare for XGBoost Training ---
xgb_grid <- expand.grid(nrounds = 50,
                        max_depth = 4,
                        eta = 0.1,
                        gamma = 0,
                        colsample_bytree = 0.8,
                        min_child_weight = 1,
                        subsample = 0.8)

# Start parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

print("Starting XGBoost training...")

# --- Train XGBoost ---
xgb_model <- train(Outcome ~ ., data = trainData, method = "xgbTree", trControl = cv_control, tuneGrid = xgb_grid)

# Stop parallel processing properly
stopCluster(cl)
stopImplicitCluster()
registerDoSEQ()

print("XGBoost training complete.")

# --- XGBoost Predictions ---
xgb_pred <- predict(xgb_model, testData)  # Already class labels
xgb_conf_matrix <- confusionMatrix(xgb_pred, testData$Outcome, positive = "1")
print(xgb_conf_matrix)

# --- Error Analysis ---
misclassified <- testData %>%
  mutate(Predicted = xgb_pred) %>%
  filter(Predicted != Outcome)

head(misclassified)  # Show first few misclassified cases

# --- ROC-AUC Calculations ---
# Logistic Regression ROC-AUC
logit_roc <- roc(testData$Outcome, logit_pred)
cat("Logistic Regression ROC-AUC: ", auc(logit_roc), "\n")

# Random Forest ROC-AUC
rf_prob <- predict(rf_model, testData, type = "prob")[,2]
rf_roc <- roc(testData$Outcome, rf_prob)
cat("Random Forest ROC-AUC: ", auc(rf_roc), "\n")

# XGBoost ROC-AUC
xgb_prob <- predict(xgb_model, testData, type = "prob")[,2]
xgb_roc <- roc(testData$Outcome, xgb_prob)
cat("XGBoost ROC-AUC: ", auc(xgb_roc), "\n")

# --- Model Performance Comparison ---
model_perf <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  AUC = c(auc(logit_roc), auc(rf_roc), auc(xgb_roc))
)

ggplot(model_perf, aes(x = Model, y = AUC, fill = Model)) +
  geom_col() +
  labs(title = "Model ROC-AUC Comparison", y = "AUC Score") +
  theme_minimal()


# --- ROC Curves Plot with AUC Values ---
plot(logit_roc, col = "blue", lwd = 2, main = "ROC Curves for All Models")
lines(rf_roc, col = "darkgreen", lwd = 2)
lines(xgb_roc, col = "red", lwd = 2)

# Extract AUC values
logit_auc <- round(auc(logit_roc), 4)
rf_auc <- round(auc(rf_roc), 4)
xgb_auc <- round(auc(xgb_roc), 4)

# Add legend with AUC values
legend("bottomright",
       legend = c(
         paste("Logistic Regression (AUC =", logit_auc, ")"),
         paste("Random Forest (AUC =", rf_auc, ")"),
         paste("XGBoost (AUC =", xgb_auc, ")")
       ),
       col = c("blue", "darkgreen", "red"),
       lwd = 2)

# --- Feature Importance ---
# Random Forest
rf_imp <- varImp(rf_model)
plot(rf_imp)

# XGBoost
xgb_imp <- varImp(xgb_model, scale = FALSE)
plot(xgb_imp)

# --- SHAP Analysis for XGBoost ---
library(SHAPforxgboost)

# Convert data to matrix format as required
train_matrix <- model.matrix(Outcome ~ . -1, data = trainData)
test_matrix <- data.matrix(testData[, -ncol(testData)])     # Exclude Outcome

# Extract the trained xgboost model object
xgb_final_model <- xgb_model$finalModel

# Calculate SHAP values
shap_values <- shap.values(xgb_model = xgb_final_model, X_train = train_matrix)

# SHAP summary plot
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = train_matrix)
shap.plot.summary(shap_long)

# --- Summary Table of ROC-AUC Scores ---
model_names <- c("Logistic Regression", "Random Forest", "XGBoost")
roc_aucs <- c(auc(logit_roc), auc(rf_roc), auc(xgb_roc))

roc_summary <- data.frame(Model = model_names, ROC_AUC = round(roc_aucs, 4))
print(roc_summary)







