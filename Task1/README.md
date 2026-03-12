# Task 1: Predict Restaurant Ratings

## Objective
Build a machine learning model to predict the aggregate rating of a restaurant based on other available features in the dataset.

## Process
1. **Data Preprocessing & Cleaning**:
   - Dropped categorical columns that uniquely identify restaurants (like `Restaurant ID`, `Restaurant Name`, `Address`) or columns directly correlated/derived from the aggregate rating (like `Rating color`, `Rating text`).
   - Addressed missing values in the `Cuisines` column by replacing them with 'Unknown'.
   - Extracted a new feature, `Cuisines Count`, by counting the number of cuisines each restaurant offers.
   - Encoded all categorical variables (such as `Has Table booking`, `Has Online delivery`, `Is delivering now`, etc.) using `LabelEncoder`.

2. **Model Selection**:
   - I used a **Random Forest Regressor** since it handles non-linear relationships well, deals effectively with a mix of categorical (encoded) and numerical features, and naturally provides feature importance.

3. **Evaluation Metrics**:
   - **Mean Squared Error (MSE)**: Used to measure the average squared difference between the predicted and actual ratings. A lower MSE indicates better accuracy.
   - **R-squared (R2)**: Used to understand what proportion of the variance in the aggregate rating is explained by our model.

## Model Performance
- **Mean Squared Error (MSE):** 0.0880
- **R-squared (R2):** 0.9613

The R2 score of 0.96 (or 96%) indicates that the model fits the data exceptionally well and can reliably predict the aggregate ratings of the restaurants based on the chosen features.

## Feature Importance Interpretation
The Random Forest model calculates how much each feature contributes to the accuracy of the prediction. Based on the analysis, here are the most influential features affecting restaurant ratings:

1. **Votes (~94.7%)**: The number of votes heavily dictates the aggregate rating. This highlights how user engagement on the platform strongly correlates with the rating assigned to the establishment. Higher engagement typically equates to extreme ratings (mostly favorable).
2. **Longitude and Latitude (~3.1% combined)**: The geographical location of the restaurant plays a very subtle but noticeable role in determining its rating. Some locations inherently attract higher ratings possibly due to affluent localities or high food-competition areas.
3. **Cuisines (~1.0%)**: The type of cuisines offered matters marginally. Some cuisines might be trending or generally preferred in the demographics represented within the dataset.
4. **Average Cost for two (~0.6%)**: Surprisingly, cost has minimal effect on the aggregate rating compared to votes and location.

*Please see `feature_importance.png` for a visual representation of the feature importances.*

## Files in this Repository
- `prediction.py`: The python script used for preprocessing, training, and evaluating the model.
- `Dataset .csv`: The original dataset provided for the task.
- `feature_importance.png`: A bar chart visualizing the importance of different features.
- `README.md`: This report detailing the process and findings.
