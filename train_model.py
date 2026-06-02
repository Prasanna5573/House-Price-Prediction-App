import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Step 1: Generating realistic synthetic housing dataset...")
np.random.seed(42)
num_samples = 1500

# Generate features
sq_ft = np.random.randint(1000, 5000, size=num_samples)
bedrooms = np.random.randint(1, 6, size=num_samples)
# Limit bathrooms based on bedrooms logically
bathrooms = np.array([np.random.randint(1, min(b + 1, 5)) for b in bedrooms])
neighborhoods = np.random.choice(["Lakeside", "Downtown", "Highlands", "Suburbs", "GreenValley"], size=num_samples)
year_built = np.random.randint(1950, 2026, size=num_samples)
has_garage = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
has_pool = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
condition = np.random.randint(1, 6, size=num_samples) # 1 to 5

# Calculate house price with realistic logic
base_price = 80000
price = (
    base_price +
    (sq_ft * 165) +
    (bedrooms * 28000) +
    (bathrooms * 35000) +
    (condition * 22000) +
    (has_garage * 18000) +
    (has_pool * 35000) -
    ((2026 - year_built) * 1200)
)

# Apply neighborhood multiplier
multipliers = {
    "Lakeside": 1.45,
    "Downtown": 1.35,
    "Highlands": 1.22,
    "Suburbs": 1.00,
    "GreenValley": 0.85
}
m_array = np.array([multipliers[n] for n in neighborhoods])
price = price * m_array

# Add some random noise (standard deviation of $15,000)
noise = np.random.normal(0, 15000, size=num_samples)
price = price + noise

# Cap minimum price to $50,000
price = np.clip(price, 50000, None)

# Assemble DataFrame
df = pd.DataFrame({
    "SquareFeet": sq_ft,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Neighborhood": neighborhoods,
    "YearBuilt": year_built,
    "HasGarage": has_garage,
    "HasPool": has_pool,
    "Condition": condition,
    "Price": price
})

# Save to CSV
df.to_csv("data/housing_data.csv", index=False)
print(f"Generated {num_samples} house records and saved to 'data/housing_data.csv'.")

print("\nStep 2: Feature Engineering & Preprocessing...")
# Calculate features
df['HouseAge'] = 2026 - df['YearBuilt']
df['SqFtPerRoom'] = df['SquareFeet'] / (df['Bedrooms'] + df['Bathrooms'] + 0.1)

# Separate features and target
X = df.drop(columns=["Price", "YearBuilt"]) # YearBuilt is replaced by HouseAge
y = df["Price"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessors
numeric_features = ["SquareFeet", "Bedrooms", "Bathrooms", "HouseAge", "Condition", "SqFtPerRoom"]
categorical_features = ["Neighborhood", "HasGarage", "HasPool"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

print("\nStep 3: Training and evaluating different regression models...")
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
}

best_model_name = None
best_model_score = -float('inf')
best_pipeline = None
results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f" - {name:16}: R2 = {r2:.4f} | MAE = ${mae:,.2f} | RMSE = ${rmse:,.2f}")
    
    results[name] = {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }
    
    if r2 > best_model_score:
        best_model_score = r2
        best_model_name = name
        best_pipeline = pipeline

print(f"\nWinner: {best_model_name} with R2 Score of {best_model_score:.4f}")

# Extract feature importances if applicable
feature_importances = {}
if best_model_name in ["RandomForest", "GradientBoosting"]:
    regressor = best_pipeline.named_steps["regressor"]
    
    # Reconstruct the feature names post one-hot encoding
    cat_encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_feature_names
    
    importances = regressor.feature_importances_
    
    # Normalize and sort importances
    for fn, imp in zip(feature_names, importances):
        # clean up name
        clean_name = fn.replace("num__", "").replace("cat__", "")
        feature_importances[clean_name] = float(imp)
        
    # Sort descending
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
else:
    # Linear Regression coefficients
    regressor = best_pipeline.named_steps["regressor"]
    cat_encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_feature_names
    
    coeffs = regressor.coef_
    # Normalize absolute coefs for a pseudo-importance comparison
    abs_coeffs = np.abs(coeffs)
    total = np.sum(abs_coeffs)
    for fn, val in zip(feature_names, abs_coeffs / total if total > 0 else abs_coeffs):
        clean_name = fn.replace("num__", "").replace("cat__", "")
        feature_importances[clean_name] = float(val)
    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

print("Feature Importances:")
for k, v in feature_importances.items():
    print(f" - {k}: {v*100:.2f}%")

print("\nStep 4: Serializing the model and saving metadata...")
# Save preprocessor and model
with open("models/model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

# Prepare metadata JSON
metadata = {
    "model_name": best_model_name,
    "metrics": results,
    "best_metrics": results[best_model_name],
    "feature_importances": feature_importances,
    "average_price": float(df["Price"].mean()),
    "average_sqft": float(df["SquareFeet"].mean()),
    "average_age": float(df["HouseAge"].mean()),
    "price_min": float(df["Price"].min()),
    "price_max": float(df["Price"].max())
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Saved model to 'models/model.pkl' and metadata to 'models/model_metadata.json'.")
print("Training completed successfully!")
