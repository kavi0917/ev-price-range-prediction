# Electric Vehicle Price and Range Prediction Using R

R mini project: predict the **driving range (km)** and the **price (euros)** of an electric car from its specifications.

- Dataset: `EV_cars.csv` (Kaggle, originally from ev-database.org), 360 cars, 9 columns
- Models: Linear Regression, Decision Tree (rpart), Random Forest (ranger)
- Evaluation: RMSE, MAE, R-squared on an 80/20 split, plus 5-fold cross-validation

## Results (test set)

| Target | Model | RMSE | R-squared |
|---|---|---|---|
| Range | Linear Regression | 41.41 km | 0.866 |
| Range | Decision Tree | 43.43 km | 0.852 |
| Range | Random Forest | 26.56 km | 0.945 |
| Price | Linear Regression | 22,182 euros | 0.678 |
| Price | Decision Tree | 17,504 euros | 0.800 |
| Price | Random Forest | 16,272 euros | 0.827 |

Note: `Efficiency` is not used to predict `Range`, because Range is almost exactly Battery x 1000 / Efficiency (data leakage).

## How to run

1. Install R and RStudio.
2. Install the packages:
   `install.packages(c("ggplot2","corrplot","caret","ranger","rpart","dplyr"))`
3. Put `EV_cars.csv` and `ev_prediction.R` in the same folder and set it as the working directory
   (RStudio: Session > Set Working Directory > To Source File Location).
4. Run `source("ev_prediction.R")`. Plots are saved in `plots/` and tables in `results/`.

## Files

- `ev_prediction.R` : complete analysis script
- `EV_cars.csv` : dataset
- `plots/` : all figures
- `results/` : model comparison tables

Author: Kavipriyadharsan P , II MCA, Karpagam College of Engineering
