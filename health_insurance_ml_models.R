############################################################
# Health Insurance Risk Classification - Sample Dataset
# Dataset: anonymized_health_insurance_policy_sample_500.xlsx
#
# Objective:
# To demonstrate the application of machine learning models
# for health insurance risk classification into three classes:
# Low, Medium, and High.
#
# Models applied:
# 1. Decision Tree
# 2. Random Forest
# 3. XGBoost
# 4. LASSO Multinomial Regression
# 5. Support Vector Machine 

############################################################
# 1. Load required packages
############################################################

required_packages <- c(
  "readxl",
  "dplyr",
  "caret",
  "recipes",
  "themis",
  "ranger",
  "rpart",
  "rpart.plot",
  "xgboost",
  "glmnet",
  "kernlab",
  "pROC",
  "ggplot2",
  "tibble"
)

new_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]

if (length(new_packages) > 0) {
  install.packages(new_packages)
}

library(readxl)
library(dplyr)
library(caret)
library(recipes)
library(themis)
library(ranger)
library(rpart)
library(rpart.plot)
library(xgboost)
library(glmnet)
library(kernlab)
library(pROC)
library(ggplot2)
library(tibble)


############################################################
# 2. Set seed for reproducibility
############################################################

set.seed(123)
############################################################
# 3. Load and prepare the dataset
############################################################

data <- read_excel("anonymized_health_insurance_policy_sample_500.xlsx")

names(data) <- c(
  "Nr_Demeve",
  "Total_Demi",
  "Mes_Vonesa_Raportimit",
  "Mes_Vonesa_Shlyerjes",
  "Pergjithshem",
  "Specialist",
  "Kirurgji",
  "Farmaci",
  "Onkologji",
  "DiagAvancuar",
  "Dentar_Optik_Fizioterapi",
  "CheckUp",
  "Laborator",
  "Tjera",
  "Mosha",
  "Gjinia",
  "Covid19",
  "log_TotalDemi",
  "TargetClass"
)

# The target variable is kept as a categorical factor.
# Low, Medium, and High

data$TargetClass <- factor(
  data$TargetClass,
  levels = c("Low", "Medium", "High")
)

cat("\n================ Target class distribution ================\n")
print(table(data$TargetClass))

if (any(is.na(data$TargetClass))) {
  stop("TargetClass contains NA values. Check that classes are exactly: Low, Medium, High.")
}


############################################################
# 4. Define predictors and response variable
############################################################

# Total_Demi and log_TotalDemi are excluded from the models.
# They are only related to the construction of the target class.

predictor_vars <- c(
  "Nr_Demeve",
  "Mes_Vonesa_Raportimit",
  "Mes_Vonesa_Shlyerjes",
  "Pergjithshem",
  "Specialist",
  "Kirurgji",
  "Farmaci",
  "Onkologji",
  "DiagAvancuar",
  "Dentar_Optik_Fizioterapi",
  "CheckUp",
  "Laborator",
  "Tjera",
  "Mosha",
  "Gjinia",
  "Covid19"
)

X <- data[, predictor_vars]

X <- X %>%
  mutate(across(everything(), as.numeric))

y <- data$TargetClass


############################################################
# 5. Fixed stratified train/test split
############################################################

set.seed(123)

train_idx <- createDataPartition(
  y,
  p = 0.80,
  list = FALSE
)

X_train <- X[train_idx, , drop = FALSE]
X_test  <- X[-train_idx, , drop = FALSE]

y_train <- y[train_idx]
y_test  <- y[-train_idx]

train_df <- data.frame(
  X_train,
  TargetClass = y_train
)

test_df <- data.frame(X_test)

train_df$TargetClass <- factor(
  train_df$TargetClass,
  levels = c("Low", "Medium", "High")
)

y_test <- factor(
  y_test,
  levels = c("Low", "Medium", "High")
)

cat("\n================ Train distribution ================\n")
print(table(train_df$TargetClass))

cat("\n================ Test distribution ================\n")
print(table(y_test))


############################################################
# 6. SMOTE on training data only
############################################################

# SMOTE is applied only to the training set.
# Continuous variables are standardized before SMOTE#

cont_vars <- c(
  "Nr_Demeve",
  "Mosha",
  "Mes_Vonesa_Raportimit",
  "Mes_Vonesa_Shlyerjes"
)

scale_params <- lapply(cont_vars, function(v) {
  
  mu <- mean(train_df[[v]], na.rm = TRUE)
  sdv <- sd(train_df[[v]], na.rm = TRUE)
  
  if (is.na(sdv) || sdv == 0) {
    sdv <- 1
  }
  
  list(mu = mu, sd = sdv)
})

names(scale_params) <- cont_vars

train_df_scaled <- train_df

for (v in cont_vars) {
  train_df_scaled[[v]] <- (
    train_df_scaled[[v]] - scale_params[[v]]$mu
  ) / scale_params[[v]]$sd
}

set.seed(123)

rec_smote <- recipe(
  TargetClass ~ .,
  data = train_df_scaled
) %>%
  step_smote(TargetClass)

prep_smote <- prep(
  rec_smote,
  training = train_df_scaled,
  retain = TRUE
)

train_smote_scaled <- bake(
  prep_smote,
  new_data = NULL
)

train_smote_raw <- train_smote_scaled

for (v in cont_vars) {
  train_smote_raw[[v]] <- train_smote_raw[[v]] *
    scale_params[[v]]$sd + scale_params[[v]]$mu
}

train_smote_raw$TargetClass <- factor(
  train_smote_raw$TargetClass,
  levels = c("Low", "Medium", "High")
)

cat("\n================ Train distribution after SMOTE ================\n")
print(table(train_smote_raw$TargetClass))


############################################################
# 7. Helper functions for evaluation
############################################################

round_numeric <- function(df, digits = 4) {
  df %>%
    mutate(across(where(is.numeric), ~ round(.x, digits)))
}


evaluate_model <- function(model_name, true_y, pred_class, prob_mat = NULL) {
  
  true_y <- factor(true_y, levels = c("Low", "Medium", "High"))
  pred_class <- factor(pred_class, levels = levels(true_y))
  
  cm <- confusionMatrix(
    data = pred_class,
    reference = true_y
  )
  
  byc <- as.data.frame(cm$byClass)
  
  class_names <- rownames(byc)
  class_names <- gsub("Class: ", "", class_names)
  
  recall <- as.numeric(byc$Sensitivity)
  precision <- as.numeric(byc$`Pos Pred Value`)
  specificity <- as.numeric(byc$Specificity)
  balanced_acc <- as.numeric(byc$`Balanced Accuracy`)
  
  f1 <- ifelse(
    precision + recall > 0,
    2 * precision * recall / (precision + recall),
    NA
  )
  
  metrics_by_class <- data.frame(
    Model = model_name,
    Class = class_names,
    Recall = recall,
    Precision = precision,
    F1 = f1,
    Specificity = specificity,
    BalancedAccuracy = balanced_acc
  )
  
  macro_metrics <- data.frame(
    Model = model_name,
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    Macro_Recall = mean(metrics_by_class$Recall, na.rm = TRUE),
    Macro_Precision = mean(metrics_by_class$Precision, na.rm = TRUE),
    Macro_F1 = mean(metrics_by_class$F1, na.rm = TRUE),
    Macro_Specificity = mean(metrics_by_class$Specificity, na.rm = TRUE),
    Macro_BalancedAccuracy = mean(metrics_by_class$BalancedAccuracy, na.rm = TRUE)
  )
  
  type_errors <- do.call(
    rbind,
    lapply(levels(true_y), function(cl) {
      
      y_bin <- factor(
        ifelse(true_y == cl, cl, "Other"),
        levels = c("Other", cl)
      )
      
      p_bin <- factor(
        ifelse(pred_class == cl, cl, "Other"),
        levels = c("Other", cl)
      )
      
      tab <- table(p_bin, y_bin)
      
      if (!("Other" %in% rownames(tab))) {
        tab <- rbind(tab, Other = 0)
      }
      
      if (!(cl %in% rownames(tab))) {
        tab <- rbind(tab, setNames(c(0, 0), colnames(tab)))
      }
      
      if (!("Other" %in% colnames(tab))) {
        tab <- cbind(tab, Other = 0)
      }
      
      if (!(cl %in% colnames(tab))) {
        tab <- cbind(tab, setNames(c(0, 0), cl))
      }
      
      TP <- tab[cl, cl]
      FP <- tab[cl, "Other"]
      FN <- tab["Other", cl]
      TN <- tab["Other", "Other"]
      
      data.frame(
        Model = model_name,
        Class = cl,
        TP = TP,
        FP = FP,
        FN = FN,
        TN = TN,
        Type_I_FPR = ifelse((FP + TN) > 0, FP / (FP + TN), NA),
        Type_II_FNR = ifelse((FN + TP) > 0, FN / (FN + TP), NA),
        Sensitivity_TPR = ifelse((TP + FN) > 0, TP / (TP + FN), NA),
        Specificity_TNR = ifelse((TN + FP) > 0, TN / (TN + FP), NA),
        Precision_PPV = ifelse((TP + FP) > 0, TP / (TP + FP), NA),
        F1 = ifelse(
          (2 * TP + FP + FN) > 0,
          (2 * TP) / (2 * TP + FP + FN),
          NA
        )
      )
    })
  )
  
  auc_table <- NULL
  roc_list <- NULL
  
  if (!is.null(prob_mat)) {
    
    prob_mat <- as.matrix(prob_mat)
    prob_mat <- prob_mat[, levels(true_y), drop = FALSE]
    
    roc_list <- lapply(levels(true_y), function(cl) {
      
      y_bin <- factor(
        ifelse(true_y == cl, cl, "Other"),
        levels = c("Other", cl)
      )
      
      roc(
        response = y_bin,
        predictor = prob_mat[, cl],
        levels = c("Other", cl),
        direction = "auto",
        quiet = TRUE
      )
    })
    
    names(roc_list) <- levels(true_y)
    
    auc_each <- sapply(
      roc_list,
      function(x) as.numeric(auc(x))
    )
    
    auc_table <- data.frame(
      Model = model_name,
      Class = names(auc_each),
      AUC = as.numeric(auc_each)
    )
    
    auc_table <- rbind(
      auc_table,
      data.frame(
        Model = model_name,
        Class = "Macro",
        AUC = mean(auc_each, na.rm = TRUE)
      )
    )
  }
  
  list(
    confusion_matrix = cm,
    metrics_by_class = metrics_by_class,
    macro_metrics = macro_metrics,
    type_errors = type_errors,
    auc_table = auc_table,
    roc_list = roc_list
  )
}


plot_confusion_matrix <- function(model_name, true_y, pred_class) {
  
  true_y <- factor(true_y, levels = c("Low", "Medium", "High"))
  pred_class <- factor(pred_class, levels = levels(true_y))
  
  cm_tbl <- table(
    Actual = true_y,
    Predicted = pred_class
  )
  
  cm_df <- as.data.frame(cm_tbl)
  colnames(cm_df) <- c("Actual", "Predicted", "Count")
  
  total_n <- sum(cm_df$Count)
  
  cm_df <- cm_df %>%
    mutate(Percent = 100 * Count / total_n)
  
  ggplot(
    cm_df,
    aes(x = Predicted, y = Actual, fill = Count)
  ) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(
      aes(label = paste0(Count, "\n(", sprintf("%.1f", Percent), "%)")),
      size = 4
    ) +
    labs(
      title = paste("Confusion Matrix -", model_name),
      x = "Predicted class",
      y = "Actual class",
      fill = "Frequency"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid = element_blank()
    )
}


plot_roc_curves <- function(model_name, roc_list, auc_table) {
  
  if (is.null(roc_list)) {
    return(NULL)
  }
  
  plot(
    roc_list[[1]],
    main = paste("ROC Curves One-vs-Rest -", model_name)
  )
  
  if (length(roc_list) > 1) {
    for (i in 2:length(roc_list)) {
      plot(roc_list[[i]], add = TRUE)
    }
  }
  
  auc_each <- auc_table %>%
    filter(Class != "Macro")
  
  legend(
    "bottomright",
    legend = paste0(
      auc_each$Class,
      " (AUC=",
      round(auc_each$AUC, 3),
      ")"
    ),
    lty = 1,
    cex = 0.85
  )
}


plot_importance <- function(importance_df, model_name, top_n = 15) {
  
  if (is.null(importance_df) || nrow(importance_df) == 0) {
    cat("\nNo variable importance available for", model_name, "\n")
    return(NULL)
  }
  
  importance_df %>%
    arrange(desc(Importance)) %>%
    slice_head(n = top_n) %>%
    ggplot(
      aes(x = reorder(Variable, Importance), y = Importance)
    ) +
    geom_col() +
    coord_flip() +
    labs(
      title = paste("Variable Importance -", model_name),
      x = "Variable",
      y = "Importance"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      axis.text.y = element_text(face = "bold")
    )
}


############################################################
# 8. Random Forest model
############################################################

# Random Forest is an ensemble learning method based on many decision trees.
# It uses bootstrap samples and random subsets of predictors to reduce variance
# and improve predictive performance.

set.seed(123)

rf_model <- ranger(
  dependent.variable.name = "TargetClass",
  data = train_smote_raw,
  num.trees = 800,
  mtry = 5,
  splitrule = "gini",
  min.node.size = 1,
  probability = TRUE,
  importance = "impurity",
  seed = 123,
  num.threads = 1
)

rf_prob <- predict(
  rf_model,
  data = test_df
)$predictions

rf_prob <- rf_prob[, levels(y_test), drop = FALSE]

rf_pred <- colnames(rf_prob)[
  max.col(rf_prob, ties.method = "first")
]

rf_pred <- factor(
  rf_pred,
  levels = levels(y_test)
)

rf_eval <- evaluate_model(
  model_name = "Random Forest",
  true_y = y_test,
  pred_class = rf_pred,
  prob_mat = rf_prob
)

rf_importance <- data.frame(
  Variable = names(rf_model$variable.importance),
  Importance = as.numeric(rf_model$variable.importance)
)

cat("\n================ RANDOM FOREST RESULTS ================\n")
print(rf_eval$confusion_matrix)

cat("\nRandom Forest - Macro metrics:\n")
print(round_numeric(rf_eval$macro_metrics, 4))

cat("\nRandom Forest - Metrics by class:\n")
print(round_numeric(rf_eval$metrics_by_class, 4))

cat("\nRandom Forest - Type I and Type II errors:\n")
print(round_numeric(rf_eval$type_errors, 4))

cat("\nRandom Forest - AUC:\n")
print(round_numeric(rf_eval$auc_table, 4))

print(plot_confusion_matrix("Random Forest", y_test, rf_pred))
plot_roc_curves("Random Forest", rf_eval$roc_list, rf_eval$auc_table)
print(plot_importance(rf_importance, "Random Forest", top_n = 15))


############################################################
# 9. Decision Tree model
############################################################

# Decision Tree is a rule-based classification method.
# It recursively splits the data into homogeneous groups.
# The model is easy to interpret but can be less stable than ensemble methods.

set.seed(123)

dt_model <- rpart(
  TargetClass ~ .,
  data = train_smote_raw,
  method = "class",
  control = rpart.control(
    cp = 0.005,
    minsplit = 20,
    minbucket = 7,
    maxdepth = 12,
    xval = 0
  )
)

dt_prob <- predict(
  dt_model,
  newdata = test_df,
  type = "prob"
)

dt_prob <- dt_prob[, levels(y_test), drop = FALSE]

dt_pred <- colnames(dt_prob)[
  max.col(dt_prob, ties.method = "first")
]

dt_pred <- factor(
  dt_pred,
  levels = levels(y_test)
)

dt_eval <- evaluate_model(
  model_name = "Decision Tree",
  true_y = y_test,
  pred_class = dt_pred,
  prob_mat = dt_prob
)

if (!is.null(dt_model$variable.importance)) {
  dt_importance <- data.frame(
    Variable = names(dt_model$variable.importance),
    Importance = as.numeric(dt_model$variable.importance)
  )
} else {
  dt_importance <- data.frame(
    Variable = character(),
    Importance = numeric()
  )
}

cat("\n================ DECISION TREE RESULTS ================\n")
print(dt_eval$confusion_matrix)

cat("\nDecision Tree - Macro metrics:\n")
print(round_numeric(dt_eval$macro_metrics, 4))

cat("\nDecision Tree - Metrics by class:\n")
print(round_numeric(dt_eval$metrics_by_class, 4))

cat("\nDecision Tree - Type I and Type II errors:\n")
print(round_numeric(dt_eval$type_errors, 4))

cat("\nDecision Tree - AUC:\n")
print(round_numeric(dt_eval$auc_table, 4))

rpart.plot(
  dt_model,
  type = 4,
  extra = 104,
  fallen.leaves = TRUE,
  tweak = 1.0,
  cex = 0.55,
  main = "Decision Tree - Final Model"
)

print(plot_confusion_matrix("Decision Tree", y_test, dt_pred))
plot_roc_curves("Decision Tree", dt_eval$roc_list, dt_eval$auc_table)
print(plot_importance(dt_importance, "Decision Tree", top_n = 15))


############################################################
# 10. XGBoost model
############################################################

# XGBoost is a gradient boosting method.
# It builds trees sequentially, where each new tree corrects the errors
# of the previous trees. It is often powerful for structured data.

best_xgb <- data.frame(
  nrounds = 600,
  max_depth = 6,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 0.7
)

ctrl_none_xgb <- trainControl(
  method = "none",
  classProbs = TRUE
)

set.seed(123)

xgb_model <- train(
  TargetClass ~ .,
  data = train_smote_raw,
  method = "xgbTree",
  trControl = ctrl_none_xgb,
  tuneGrid = best_xgb,
  metric = "Accuracy",
  verbose = FALSE,
  nthread = 1
)

xgb_prob <- predict(
  xgb_model,
  newdata = test_df,
  type = "prob"
)

xgb_prob <- xgb_prob[, levels(y_test), drop = FALSE]

xgb_pred <- colnames(xgb_prob)[
  max.col(xgb_prob, ties.method = "first")
]

xgb_pred <- factor(
  xgb_pred,
  levels = levels(y_test)
)

xgb_eval <- evaluate_model(
  model_name = "XGBoost",
  true_y = y_test,
  pred_class = xgb_pred,
  prob_mat = xgb_prob
)

xgb_importance <- varImp(
  xgb_model,
  scale = FALSE
)$importance %>%
  rownames_to_column("Variable") %>%
  rename(Importance = Overall)

cat("\n================ XGBOOST RESULTS ================\n")
print(xgb_eval$confusion_matrix)

cat("\nXGBoost - Macro metrics:\n")
print(round_numeric(xgb_eval$macro_metrics, 4))

cat("\nXGBoost - Metrics by class:\n")
print(round_numeric(xgb_eval$metrics_by_class, 4))

cat("\nXGBoost - Type I and Type II errors:\n")
print(round_numeric(xgb_eval$type_errors, 4))

cat("\nXGBoost - AUC:\n")
print(round_numeric(xgb_eval$auc_table, 4))

print(plot_confusion_matrix("XGBoost", y_test, xgb_pred))
plot_roc_curves("XGBoost", xgb_eval$roc_list, xgb_eval$auc_table)
print(plot_importance(xgb_importance, "XGBoost", top_n = 15))


############################################################
# 11. LASSO Multinomial Regression
############################################################

# LASSO Multinomial Regression is a regularized classification model.
# LASSO applies an L1 penalty, which can shrink some coefficients to zero.
# This makes the model useful for variable selection and interpretability.

set.seed(123)

x_train_lasso <- model.matrix(
  ~ . - 1,
  data = train_smote_raw %>% select(-TargetClass)
)

y_train_lasso <- train_smote_raw$TargetClass

x_test_lasso <- model.matrix(
  ~ . - 1,
  data = test_df
)

common_cols <- intersect(
  colnames(x_train_lasso),
  colnames(x_test_lasso)
)

x_train_lasso <- x_train_lasso[, common_cols, drop = FALSE]
x_test_lasso  <- x_test_lasso[, common_cols, drop = FALSE]

lasso_lambda <- 2.872826e-03

lasso_model <- glmnet(
  x = x_train_lasso,
  y = y_train_lasso,
  family = "multinomial",
  alpha = 1,
  lambda = lasso_lambda
)

lasso_prob <- predict(
  lasso_model,
  newx = x_test_lasso,
  type = "response"
)[,,1]

colnames(lasso_prob) <- levels(y_test)

lasso_prob <- lasso_prob[, levels(y_test), drop = FALSE]

lasso_pred <- colnames(lasso_prob)[
  max.col(lasso_prob, ties.method = "first")
]

lasso_pred <- factor(
  lasso_pred,
  levels = levels(y_test)
)

lasso_eval <- evaluate_model(
  model_name = "LASSO Multinomial",
  true_y = y_test,
  pred_class = lasso_pred,
  prob_mat = lasso_prob
)

lasso_coef_list <- coef(lasso_model)

lasso_all_coefficients <- bind_rows(
  lapply(names(lasso_coef_list), function(cl) {
    
    mat <- as.matrix(lasso_coef_list[[cl]])
    
    data.frame(
      Class = cl,
      Variable = rownames(mat),
      Coef = as.numeric(mat[, 1])
    )
  })
) %>%
  filter(Variable != "(Intercept)")

lasso_nonzero_coefficients <- lasso_all_coefficients %>%
  filter(Coef != 0) %>%
  arrange(Class, desc(abs(Coef)))

lasso_importance <- lasso_all_coefficients %>%
  group_by(Variable) %>%
  summarise(
    Importance = sum(abs(Coef)),
    .groups = "drop"
  ) %>%
  arrange(desc(Importance))

cat("\n================ LASSO MULTINOMIAL RESULTS ================\n")
print(lasso_eval$confusion_matrix)

cat("\nLASSO - Macro metrics:\n")
print(round_numeric(lasso_eval$macro_metrics, 4))

cat("\nLASSO - Metrics by class:\n")
print(round_numeric(lasso_eval$metrics_by_class, 4))

cat("\nLASSO - Type I and Type II errors:\n")
print(round_numeric(lasso_eval$type_errors, 4))

cat("\nLASSO - AUC:\n")
print(round_numeric(lasso_eval$auc_table, 4))

cat("\nLASSO - Non-zero coefficients:\n")
print(lasso_nonzero_coefficients)

print(plot_confusion_matrix("LASSO Multinomial", y_test, lasso_pred))
plot_roc_curves("LASSO Multinomial", lasso_eval$roc_list, lasso_eval$auc_table)

############################################################
# 12. Support Vector Machine with RBF Kernel
############################################################

# SVM with RBF kernel is a nonlinear classification model.
# The RBF kernel allows the model to capture nonlinear decision boundaries.

set.seed(123)

preproc_svm <- preProcess(
  X_train[, cont_vars, drop = FALSE],
  method = c("center", "scale")
)

X_train_svm <- X_train
X_test_svm  <- X_test

X_train_svm[, cont_vars] <- predict(
  preproc_svm,
  X_train[, cont_vars, drop = FALSE]
)

X_test_svm[, cont_vars] <- predict(
  preproc_svm,
  X_test[, cont_vars, drop = FALSE]
)

train_df_svm <- data.frame(
  X_train_svm,
  TargetClass = y_train
)

train_df_svm$TargetClass <- factor(
  train_df_svm$TargetClass,
  levels = c("Low", "Medium", "High")
)

set.seed(123)

rec_smote_svm <- recipe(
  TargetClass ~ .,
  data = train_df_svm
) %>%
  step_smote(TargetClass)

prep_smote_svm <- prep(
  rec_smote_svm,
  training = train_df_svm,
  retain = TRUE
)

train_smote_svm <- bake(
  prep_smote_svm,
  new_data = NULL
)

train_smote_svm$TargetClass <- factor(
  train_smote_svm$TargetClass,
  levels = c("Low", "Medium", "High")
)

x_train_svm <- train_smote_svm %>%
  select(-TargetClass)

y_train_svm <- train_smote_svm$TargetClass

dv_svm <- dummyVars(
  ~ .,
  data = x_train_svm,
  fullRank = TRUE
)

x_train_svm_num <- as.data.frame(
  predict(dv_svm, newdata = x_train_svm)
)

x_test_svm_num <- as.data.frame(
  predict(dv_svm, newdata = X_test_svm)
)

x_test_svm_num <- x_test_svm_num[
  ,
  colnames(x_train_svm_num),
  drop = FALSE
]

nzv <- nearZeroVar(x_train_svm_num)

if (length(nzv) > 0) {
  x_train_svm_num <- x_train_svm_num[, -nzv, drop = FALSE]
  x_test_svm_num  <- x_test_svm_num[, -nzv, drop = FALSE]
}

set.seed(123)

svm_model <- ksvm(
  x = as.matrix(x_train_svm_num),
  y = y_train_svm,
  type = "C-svc",
  kernel = "rbfdot",
  kpar = list(sigma = 0.01),
  C = 10,
  scaled = FALSE,
  prob.model = TRUE
)

svm_pred <- predict(
  svm_model,
  newdata = as.matrix(x_test_svm_num),
  type = "response"
)

svm_pred <- factor(
  svm_pred,
  levels = levels(y_test)
)

svm_prob <- predict(
  svm_model,
  newdata = as.matrix(x_test_svm_num),
  type = "probabilities"
)

svm_prob <- as.matrix(svm_prob)
svm_prob <- svm_prob[, levels(y_test), drop = FALSE]

svm_eval <- evaluate_model(
  model_name = "SVM RBF",
  true_y = y_test,
  pred_class = svm_pred,
  prob_mat = svm_prob
)

cat("\n================ SVM RESULTS ================\n")
print(svm_eval$confusion_matrix)

cat("\nSVM - Macro metrics:\n")
print(round_numeric(svm_eval$macro_metrics, 4))

cat("\nSVM - Metrics by class:\n")
print(round_numeric(svm_eval$metrics_by_class, 4))

cat("\nSVM - Type I and Type II errors:\n")
print(round_numeric(svm_eval$type_errors, 4))

cat("\nSVM - AUC:\n")
print(round_numeric(svm_eval$auc_table, 4))

print(plot_confusion_matrix("SVM RBF", y_test, svm_pred))
plot_roc_curves("SVM RBF", svm_eval$roc_list, svm_eval$auc_table)
print(plot_importance(svm_importance, "SVM RBF", top_n = 15))


############################################################
# 13. Final comparison of all models
############################################################

# The final comparison tables summarize model performance
# using the same test set for all models.

all_macro_metrics <- bind_rows(
  rf_eval$macro_metrics,
  dt_eval$macro_metrics,
  xgb_eval$macro_metrics,
  lasso_eval$macro_metrics,
  svm_eval$macro_metrics
)

all_class_metrics <- bind_rows(
  rf_eval$metrics_by_class,
  dt_eval$metrics_by_class,
  xgb_eval$metrics_by_class,
  lasso_eval$metrics_by_class,
  svm_eval$metrics_by_class
)

all_type_errors <- bind_rows(
  rf_eval$type_errors,
  dt_eval$type_errors,
  xgb_eval$type_errors,
  lasso_eval$type_errors,
  svm_eval$type_errors
)

all_auc <- bind_rows(
  rf_eval$auc_table,
  dt_eval$auc_table,
  xgb_eval$auc_table,
  lasso_eval$auc_table,
  svm_eval$auc_table
)

cat("\n================ FINAL MODEL COMPARISON: MACRO METRICS ================\n")
print(round_numeric(all_macro_metrics, 4))

############################################################
# 14. High-risk class performance comparison
############################################################

# In insurance risk classification, the High class is the most important class.
# A false negative for the High class means that a high-risk policyholder
# is classified as Low or Medium risk.

high_class_metrics <- bind_rows(
  rf_eval$metrics_by_class,
  dt_eval$metrics_by_class,
  xgb_eval$metrics_by_class,
  lasso_eval$metrics_by_class,
  svm_eval$metrics_by_class
) %>%
  filter(Class == "High") %>%
  select(
    Model,
    Class,
    Recall,
    Precision,
    F1,
    Specificity,
    BalancedAccuracy
  )

high_class_type_errors <- bind_rows(
  rf_eval$type_errors,
  dt_eval$type_errors,
  xgb_eval$type_errors,
  lasso_eval$type_errors,
  svm_eval$type_errors
) %>%
  filter(Class == "High") %>%
  select(
    Model,
    Class,
    TP,
    FP,
    FN,
    TN,
    Type_I_FPR,
    Type_II_FNR,
    Sensitivity_TPR,
    Specificity_TNR,
    Precision_PPV,
    F1
  )

high_class_auc <- bind_rows(
  rf_eval$auc_table,
  dt_eval$auc_table,
  xgb_eval$auc_table,
  lasso_eval$auc_table,
  svm_eval$auc_table
) %>%
  filter(Class == "High") %>%
  select(
    Model,
    Class,
    AUC
  )

high_class_comparison <- high_class_metrics %>%
  left_join(
    high_class_type_errors,
    by = c("Model", "Class"),
    suffix = c("_metric", "_error")
  ) %>%
  left_join(
    high_class_auc,
    by = c("Model", "Class")
  ) %>%
  select(
    Model,
    Class,
    TP,
    FP,
    FN,
    TN,
    Recall,
    Precision,
    F1_metric,
    Specificity,
    BalancedAccuracy,
    Type_I_FPR,
    Type_II_FNR,
    AUC
  ) %>%
  arrange(desc(Recall), desc(F1_metric), desc(AUC))

cat("\n================ HIGH CLASS METRICS COMPARISON ================\n")
print(round_numeric(high_class_comparison, 4))


##########################  END  ######################################################
