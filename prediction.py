import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting Restaurant Ratings Prediction Pipeline...")

# 1. Load the dataset
df = pd.read_csv('Dataset .csv')
print(f"Dataset loaded. Shape: {df.shape}")

# 2. Preprocess the dataset
# Drop columns that are intuitively not helpful for broad prediction or would leak the target
# 'Rating color' and 'Rating text' are just categories based on 'Aggregate rating'
cols_to_drop = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'City', 'Address', 
                'Locality', 'Locality Verbose', 'Currency', 'Rating color', 'Rating text']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Handle missing values
# Check missing
print("\nMissing values before handling:")
print(df.isna().sum())

# Cuisines has some missing values. We will fill them with 'Unknown'
df.fillna({'Cuisines': 'Unknown'}, inplace=True)
df.dropna(inplace=True)

# Feature Engineering / Encoding
# Count the number of cuisines offered by the restaurant
df['Cuisines Count'] = df['Cuisines'].apply(lambda x: len(x.split(',')))

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define Features and Target
X = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into training and testing sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 3. Model Training
print("\nTraining Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. Evaluation
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# 5. Interpret Model Results / Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances in Predicting Aggregate Rating')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nSaved feature_importance.png")
