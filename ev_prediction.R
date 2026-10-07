# =====================================================================
#  Electric Vehicle Price and Range Prediction using R
#  Dataset : EV_cars.csv (Kaggle / ev-database.org, 360 cars)
#  Models  : Linear Regression, Decision Tree, Random Forest
#  Targets : (A) Range in km      (B) Price in euros
# =====================================================================

# ---- 0. Packages ----------------------------------------------------
# install.packages(c("ggplot2","corrplot","caret","ranger","rpart","dplyr"))
library(ggplot2)
library(corrplot)
library(caret)
library(ranger)
library(rpart)
library(dplyr)

dir.create("plots",   showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# ---- 1. Load the dataset --------------------------------------------
ev <- read.csv("EV_cars.csv", stringsAsFactors = FALSE)
head(ev)
dim(ev)
names(ev)
str(ev)
summary(ev)

# ---- 2. Stage 1 : Data cleaning -------------------------------------
# 2.1 Give the awkward column names simple names
names(ev)[names(ev) == "Price.DE."]            <- "Price"
names(ev)[names(ev) == "acceleration..0.100."] <- "Acceleration"

# 2.2 Missing values
na_summary <- colSums(is.na(ev))
print(na_summary)

# 2.3 Duplicate rows
dup_count <- sum(duplicated(ev))
cat("Duplicate rows:", dup_count, "\n")

# 2.4 Remove the web-link column (it is only text, not a feature)
ev$Car_name_link <- NULL

# 2.5 Fast_charge has only 2 missing values -> fill with the median
ev$Fast_charge[is.na(ev$Fast_charge)] <- median(ev$Fast_charge, na.rm = TRUE)

# 2.6 Price has many missing values (it is the thing we want to predict
#     in Model B). Filling a target with the median would create fake
#     information, so Model B uses only cars whose price is known.
cat("Cars with missing Price:", sum(is.na(ev$Price)), "\n")
ev_price <- ev[!is.na(ev$Price), ]
cat("Cars available for the Price model:", nrow(ev_price), "\n")

# 2.7 Outlier check with the IQR rule
count_outliers <- function(x) {
  q <- quantile(x, c(0.25, 0.75), na.rm = TRUE)
  iqr <- q[2] - q[1]
  sum(x < q[1] - 1.5 * iqr | x > q[2] + 1.5 * iqr, na.rm = TRUE)
}
num_cols <- c("Battery","Efficiency","Fast_charge","Price","Range",
              "Top_speed","Acceleration")
outlier_table <- data.frame(
  Feature  = num_cols,
  Outliers = sapply(ev[, num_cols], count_outliers)
)
print(outlier_table)
# Outliers are real cars (luxury / performance EVs), not errors,
# so they are kept.

# ---- 3. Feature engineering -----------------------------------------
# Brand = first word of the car name (used for exploration)
ev$Brand <- sapply(strsplit(ev$Car_name, " "), `[`, 1)
ev$Price_per_km <- ev$Price / ev$Range      # euros paid for each km of range
ev_price <- ev[!is.na(ev$Price), ]           # refresh so it also has Brand and Price_per_km

# ---- 4. Stage 2 : Data visualization --------------------------------
theme_set(theme_minimal(base_size = 13))

p1 <- ggplot(ev_price, aes(x = Price)) +
  geom_histogram(bins = 30, fill = "#2a7f62", colour = "white") +
  ggtitle("Distribution of Price (euros)")
ggsave("plots/01_price_hist.png", p1, width = 7, height = 4.5, dpi = 150)

p2 <- ggplot(ev, aes(x = Range)) +
  geom_histogram(bins = 30, fill = "#1f6f8b", colour = "white") +
  ggtitle("Distribution of Range (km)")
ggsave("plots/02_range_hist.png", p2, width = 7, height = 4.5, dpi = 150)

num_vars <- ev[, num_cols]
cor_mat  <- cor(num_vars, use = "pairwise.complete.obs")
png("plots/03_correlation.png", width = 900, height = 800, res = 130)
corrplot(cor_mat, method = "color", addCoef.col = "black",
         number.cex = 0.7, tl.cex = 0.8, tl.col = "black")
dev.off()
round(cor_mat, 2)

# Boxplots (each feature has its own scale, so facet with free y-axis)
long <- stack(ev[, c("Battery","Efficiency","Fast_charge","Range",
                     "Top_speed","Acceleration")])
p4 <- ggplot(long, aes(x = ind, y = values)) +
  geom_boxplot(fill = "#e8f1ee") +
  facet_wrap(~ ind, scales = "free") +
  labs(x = NULL, y = NULL) + ggtitle("Boxplots for checking outliers") +
  theme(axis.text.x = element_blank())
ggsave("plots/04_boxplots.png", p4, width = 9, height = 5.5, dpi = 150)

p5 <- ggplot(ev, aes(x = Battery, y = Range)) +
  geom_point(alpha = 0.6, colour = "#1f6f8b") +
  geom_smooth(method = "lm", colour = "#c0392b", se = FALSE) +
  ggtitle("Battery capacity vs Range")
ggsave("plots/05_battery_vs_range.png", p5, width = 7, height = 4.5, dpi = 150)

p6 <- ggplot(ev_price, aes(x = Top_speed, y = Price)) +
  geom_point(alpha = 0.6, colour = "#2a7f62") +
  geom_smooth(method = "lm", colour = "#c0392b", se = FALSE) +
  ggtitle("Top speed vs Price")
ggsave("plots/06_speed_vs_price.png", p6, width = 7, height = 4.5, dpi = 150)

brand_price <- ev_price %>% group_by(Brand) %>%
  summarise(n = n(), AvgPrice = mean(Price)) %>%
  filter(n >= 5) %>% arrange(desc(AvgPrice)) %>% head(10)
p7 <- ggplot(brand_price, aes(x = reorder(Brand, AvgPrice), y = AvgPrice)) +
  geom_col(fill = "#2a7f62") + coord_flip() +
  labs(x = NULL, y = "Average price (euros)") +
  ggtitle("Top 10 brands by average price (brands with 5+ models)")
ggsave("plots/07_brand_price.png", p7, width = 7, height = 4.5, dpi = 150)

# ---- 5. Stage 3 : Important note about Range --------------------------
# Range is almost exactly Battery*1000/Efficiency, so using Efficiency
# to predict Range would be like giving the answer away (data leakage).
leak <- cor(ev$Range * ev$Efficiency / 1000, ev$Battery)
cat("Correlation of Range*Efficiency/1000 with Battery:", round(leak, 5), "\n")

# ---- 6. Stage 4 : Data splitting (80% train, 20% test) --------------
set.seed(42)
idxA   <- createDataPartition(ev$Range, p = 0.8, list = FALSE)
trainA <- ev[idxA, ];  testA <- ev[-idxA, ]
cat("Range model  -> Train rows:", nrow(trainA), "| Test rows:", nrow(testA), "\n")

idxB   <- createDataPartition(ev_price$Price, p = 0.8, list = FALSE)
trainB <- ev_price[idxB, ];  testB <- ev_price[-idxB, ]
cat("Price model  -> Train rows:", nrow(trainB), "| Test rows:", nrow(testB), "\n")

featA <- c("Battery","Fast_charge","Top_speed","Acceleration")
featB <- c("Battery","Efficiency","Fast_charge","Top_speed","Acceleration","Range")

fA <- as.formula(paste("Range ~",  paste(featA, collapse = " + ")))
fB <- as.formula(paste("Price ~",  paste(featB, collapse = " + ")))

# helper: evaluation metrics
evaluate <- function(actual, pred) {
  rmse <- sqrt(mean((actual - pred)^2))
  mae  <- mean(abs(actual - pred))
  r2   <- 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
  c(RMSE = rmse, MAE = mae, R2 = r2)
}

# ---- 7. Stage 5 : Model training ------------------------------------
# ===== MODEL A : predict RANGE =====
lmA <- lm(fA, data = trainA)
summary(lmA)
dtA <- rpart(fA, data = trainA, method = "anova")
rfA <- ranger(fA, data = trainA, num.trees = 300,
              importance = "permutation", seed = 42)

predA <- data.frame(
  Actual = testA$Range,
  LR = predict(lmA, testA),
  DT = predict(dtA, testA),
  RF = predict(rfA, testA)$predictions
)
resA <- rbind(
  `Linear Regression` = evaluate(predA$Actual, predA$LR),
  `Decision Tree`     = evaluate(predA$Actual, predA$DT),
  `Random Forest`     = evaluate(predA$Actual, predA$RF)
)
cat("\n--- Range prediction (test set) ---\n"); print(round(resA, 3))

# ===== MODEL B : predict PRICE =====
lmB <- lm(fB, data = trainB)
summary(lmB)
dtB <- rpart(fB, data = trainB, method = "anova")
rfB <- ranger(fB, data = trainB, num.trees = 300,
              importance = "permutation", seed = 42)

predB <- data.frame(
  Actual = testB$Price,
  LR = predict(lmB, testB),
  DT = predict(dtB, testB),
  RF = predict(rfB, testB)$predictions
)
resB <- rbind(
  `Linear Regression` = evaluate(predB$Actual, predB$LR),
  `Decision Tree`     = evaluate(predB$Actual, predB$DT),
  `Random Forest`     = evaluate(predB$Actual, predB$RF)
)
cat("\n--- Price prediction (test set) ---\n"); print(round(resB, 3))

write.csv(round(resA, 4), "results/range_model_comparison.csv")
write.csv(round(resB, 4), "results/price_model_comparison.csv")

# ---- 8. Actual vs Predicted plots -----------------------------------
avp_plot <- function(df, title, unit) {
  long <- rbind(
    data.frame(Model = "Linear Regression", Actual = df$Actual, Predicted = df$LR),
    data.frame(Model = "Decision Tree",     Actual = df$Actual, Predicted = df$DT),
    data.frame(Model = "Random Forest",     Actual = df$Actual, Predicted = df$RF))
  ggplot(long, aes(Actual, Predicted)) +
    geom_point(alpha = 0.6, colour = "#1f6f8b") +
    geom_abline(slope = 1, intercept = 0, colour = "#c0392b") +
    facet_wrap(~ Model) +
    labs(x = paste("Observed", unit), y = paste("Predicted", unit)) +
    ggtitle(title)
}
ggsave("plots/08_range_actual_vs_pred.png",
       avp_plot(predA, "Range: Predicted vs Observed", "range (km)"),
       width = 10, height = 3.6, dpi = 150)
ggsave("plots/09_price_actual_vs_pred.png",
       avp_plot(predB, "Price: Predicted vs Observed", "price (euros)"),
       width = 10, height = 3.6, dpi = 150)

# ---- 9. Feature importance (Random Forest) --------------------------
imp_plot <- function(model, title) {
  imp <- sort(model$variable.importance)
  d <- data.frame(Feature = factor(names(imp), levels = names(imp)),
                  Importance = as.numeric(imp))
  ggplot(d, aes(Feature, Importance)) + geom_col(fill = "#2a7f62") +
    coord_flip() + ggtitle(title)
}
ggsave("plots/10_range_importance.png",
       imp_plot(rfA, "Random Forest feature importance: Range"),
       width = 7, height = 4, dpi = 150)
ggsave("plots/11_price_importance.png",
       imp_plot(rfB, "Random Forest feature importance: Price"),
       width = 7, height = 4, dpi = 150)
print(sort(rfA$variable.importance, decreasing = TRUE))
print(sort(rfB$variable.importance, decreasing = TRUE))

# ---- 10. 5-fold cross-validation (checks the result is stable) ------
ctrl <- trainControl(method = "cv", number = 5)
set.seed(42)
cvA <- train(fA, data = ev, method = "lm", trControl = ctrl)
cvA_rf <- train(fA, data = ev, method = "ranger", trControl = ctrl,
                tuneGrid = data.frame(mtry = 2, splitrule = "variance",
                                      min.node.size = 5))
cvB <- train(fB, data = ev_price, method = "lm", trControl = ctrl)
cvB_rf <- train(fB, data = ev_price, method = "ranger", trControl = ctrl,
                tuneGrid = data.frame(mtry = 3, splitrule = "variance",
                                      min.node.size = 5))
cv_table <- data.frame(
  Target = c("Range","Range","Price","Price"),
  Model  = c("Linear Regression","Random Forest",
             "Linear Regression","Random Forest"),
  CV_RMSE = c(cvA$results$RMSE, cvA_rf$results$RMSE,
              cvB$results$RMSE, cvB_rf$results$RMSE),
  CV_R2   = c(cvA$results$Rsquared, cvA_rf$results$Rsquared,
              cvB$results$Rsquared, cvB_rf$results$Rsquared))
print(cv_table)
write.csv(cv_table, "results/cross_validation.csv", row.names = FALSE)

# ---- 11. Demo: predict for a new, imaginary car ---------------------
new_car <- data.frame(Battery = 75, Efficiency = 170, Fast_charge = 600,
                      Top_speed = 200, Acceleration = 6.5, Range = 420)
cat("\nPredicted range (km):",
    round(predict(rfA, new_car)$predictions, 1), "\n")
cat("Predicted price (euros):",
    round(predict(rfB, new_car)$predictions, 0), "\n")

cat("\nDONE. Plots are in /plots and tables are in /results\n")
