# Install and load all required packages

required_packages <- c("caTools", "caret", "dplyr", "smotefamily", "xgboost", "benford.analysis", "pROC", "ggplot2", "knitr")

install_and_load <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

lapply(required_packages, install_and_load)

# Exploratory data analysis
# Read the CSV file in working directory
data <- read_csv("creditcard.csv")
# Inspect the data
head(data)
str(data)
summary(data)

# Check for missing values
missing_values <- sapply(data, function(x) sum(is.na(x)))
missing_values

# Check the distribution of the target variable, Class, for by counting 
# the number of fraud and non-fraud transactions

class_counts <- data %>%
  count(Class)
print(class_counts)

# Features V1 to V28 are unnamed and reportedly result from PCA analysis of the 
# original data.  If so, the standard deviations of these features should decrease as the feature 
# number increases.  Evaluating here to confirm this.

# Compute standard deviations for each feature
sds <- sapply(data[, paste0("V", 1:28)], sd)
  
# Look at the distribution of SDs for each feature
# Compute standard deviations for each feature
sds <- sapply(data[, paste0("V", 1:28)], sd)

# Create a data frame for the standard deviations
sds_df <- data.frame(
  Feature = paste0("V", 1:28),
  SD = sds
)
# Convert Feature to a factor with levels in the correct order
sds_df$Feature <- factor(sds_df$Feature, levels = paste0("V", 1:28))

# Inspect the data frame
print(sds_df)

# Plot the standard deviations
ggplot(sds_df, aes(x = Feature, y = SD)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(SD, 2)), vjust = -0.5, size = 3.5) +
  labs(title = "Standard Deviations of Features V1 to V28",
       x = "Feature",
       y = "Standard Deviation") +
  theme_linedraw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# So the data is tidy, very imbalanced and the numbered features are the result 
# of PCA analysis of the original data.

# Examining relationships of Amount and class as well as Benford analysis of 
# the Amount feature

#Examine the Amount feature to see if the distribution is different for fraudulent and non-fraudulent transactions
# Create a subset of the data with only positive amounts
positive_amount_data <- data[data$Amount > 0, ]
ggplot(positive_amount_data, aes(x = Amount, fill = as.factor(Class))) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Transaction Amounts by Class",
       x = "Amount",
       y = "Density",
       fill = "Class") +
  scale_x_log10() +
  theme_minimal()

# Perform Benford's Law analysis on the Amount column
benford_results <- benford(data$Amount, number.of.digits = 2, discrete = TRUE, round = 2)
# Plot the results of the Benford's Law analysis
plot(benford_results)

#Evaluate if Benford outliers could be indicative of fraudulent transactions
# Identify outliers using the benford.analysis package
amount_as_df <- data.frame(Amount = data$Amount)
benford_outliers <- getSuspects(benford_results, amount_as_df, by = 'absolute.diff')
head(benford_outliers)

# Extract amounts considered outliers
outlier_amounts <- benford_outliers$Amount

# Create a boolean column to indicate if a transaction is an outlier
data$Outlier <- data$Amount %in% outlier_amounts

# Inspect the first few rows to verify
head(data)

# Create a contingency table of Benford outliers vs. Class as the visualization was not so useful
contingency_table <- table(data$Outlier, data$Class)

# Perform Chi-square test
chi_square_test <- chisq.test(contingency_table)

# Print the contingency table
print(contingency_table)

# Print the results of the Chi-square test
print(chi_square_test)

# The Chi-square test results indicate that the Benford outliers are significantly 
# associated with the Class variable. I will use them as a feature in the model.

# Generating the model for the credit card fraud detection
# Load the dataset once again as I altered it in the previous steps
data <- read.csv("creditcard.csv")

# Set seed for reproducibility
set.seed(123)

# Split the data
split <- sample.split(data$Class, SplitRatio = 0.7)
train_data <- subset(data, split == TRUE)
test_data <- subset(data, split == FALSE)

# Perform SMOTE on the training data to improve class balance in training
train_data_smote <- SMOTE(train_data, train_data$Class, K = 5, dup_size = 0)
balanced_train <- train_data_smote$data
str(balanced_train)
# Remove the extra 'class' column added by the SMOTE function
balanced_train <- balanced_train[, -ncol(balanced_train)]

# Check the structure to confirm
str(balanced_train)

# Check the class distribution of the balanced training data
class_counts_balanced <- balanced_train %>%
  count(Class)

# Calculate Benford Outliers for Training Set
benford_results_train <- benford(balanced_train$Amount, number.of.digits = 2)
benford_outliers_train <- getSuspects(benford_results_train, data.frame(Amount = balanced_train$Amount))
balanced_train$Benford_Outlier <- balanced_train$Amount %in% benford_outliers_train$Amount

# Calculate Benford Outliers for Testing Set
benford_results_test <- benford(test_data$Amount, number.of.digits = 2)
benford_outliers_test <- getSuspects(benford_results_test, data.frame(Amount = test_data$Amount))
test_data$Benford_Outlier <- test_data$Amount %in% benford_outliers_test$Amount
# Ensure columns are in the same order
common_cols <- intersect(names(balanced_train), names(test_data))
balanced_train <- balanced_train[, common_cols]
test_data <- test_data[, common_cols]

# Prepare data for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(balanced_train[, -which(names(balanced_train) %in% c("Class"))]), label = balanced_train$Class)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) %in% c("Class"))]), label = test_data$Class)

# Set XGBoost parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc"
)

# Train the XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix, eval = test_matrix),
  verbose = 1
)

# Evaluate the model
preds <- predict(xgb_model, test_matrix)
roc_auc <- roc(test_data$Class, preds)

# Calculate confusion matrix and additional metrics
preds_binary <- ifelse(preds > 0.5, 1, 0) # Convert probabilities to binary predictions
conf_matrix <- confusionMatrix(as.factor(preds_binary), as.factor(test_data$Class))
precision <- posPredValue(as.factor(preds_binary), as.factor(test_data$Class))
recall <- sensitivity(as.factor(preds_binary), as.factor(test_data$Class))
f1_score <- (2 * precision * recall) / (precision + recall)

# Print results
cat("AUC: ", roc_auc$auc, "\n")
print(conf_matrix)
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")

# Feature importance
importance_matrix <- xgb.importance(model = xgb_model)
# Plot using xgboost's built-in function
xgb.plot.importance(importance_matrix, top_n = 32)  

# Select top features for a simplified model
top_features <- importance_matrix$Feature[1:15]

# Subset data to only include top features
train_data_top <- balanced_train[, c("Class", top_features)]
test_data_top <- test_data[, c("Class", top_features)]

# Prepare data for XGBoost with top features
train_matrix_top <- xgb.DMatrix(data = as.matrix(train_data_top[, -which(names(train_data_top) %in% c("Class"))]), label = train_data_top$Class)
test_matrix_top <- xgb.DMatrix(data = as.matrix(test_data_top[, -which(names(test_data_top) %in% c("Class"))]), label = test_data_top$Class)

# Train the XGBoost model with top features
xgb_model_top <- xgb.train(
  params = params,
  data = train_matrix_top,
  nrounds = 100,
  watchlist = list(train = train_matrix_top, eval = test_matrix_top),
  verbose = 1
)

# Evaluate the model with top features
preds_top <- predict(xgb_model_top, test_matrix_top)
roc_auc_top <- roc(test_data_top$Class, preds_top)

# Calculate confusion matrix and additional metrics for top features model
preds_binary_top <- ifelse(preds_top > 0.5, 1, 0)
conf_matrix_top <- confusionMatrix(as.factor(preds_binary_top), as.factor(test_data_top$Class))
precision_top <- posPredValue(as.factor(preds_binary_top), as.factor(test_data_top$Class))
recall_top <- sensitivity(as.factor(preds_binary_top), as.factor(test_data_top$Class))
f1_score_top <- (2 * precision_top * recall_top) / (precision_top + recall_top)

# Print results for top features model
cat("AUC (Top Features): ", roc_auc_top$auc, "\n")
print(conf_matrix_top)
cat("Precision (Top Features): ", precision_top, "\n")
cat("Recall (Top Features): ", recall_top, "\n")
cat("F1 Score (Top Features): ", f1_score_top, "\n")

# Calculate costs
total_fraud_amount <- sum(test_data$Amount[test_data$Class == 1])
missed_fraud_amount_original <- sum(test_data$Amount[(test_data$Class == 1) & (preds_binary == 0)])
missed_fraud_amount_top <- sum(test_data$Amount[(test_data$Class == 1) & (preds_binary_top == 0)])
deployment_cost <- 5000  # Hypothetical deployment cost
false_positive_cost <- 100  # Hypothetical cost per false positive
false_positives_original <- sum((test_data_top$Class == 0) & (preds_binary == 1))
false_positives_top <- sum((test_data_top$Class == 0) & (preds_binary_top == 1))

original_model_cost <- missed_fraud_amount_original + deployment_cost + (false_positives_original * false_positive_cost)
top_features_model_cost <- missed_fraud_amount_top + deployment_cost + (false_positives_top * false_positive_cost)

# Print cost comparison to determine best detection strategy to deploy
cat("Total Fraud Amount (No Model): $", total_fraud_amount, "\n")
cat("Original Model Cost: $", original_model_cost, "\n")
cat("Top Features Model Cost: $", top_features_model_cost, "\n")
