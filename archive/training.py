import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOADING THE DATA
df = pd.read_csv("cleaned_car_data.csv")

# 2. DEFINING FEATURES (X) AND TARGET (y)
X = df.drop('price', axis=1)
y = df['price']

# 3. SPLITTING THE DATA (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} cars, Testing on {len(X_test)} cars.")

# 4. PREPROCESSING PIPELINE
categorical_cols = ['brand', 'transmission', 'fuelType'] 
numerical_cols = ['mileage', 'tax', 'mpg', 'engineSize', 'car_age']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# MODEL 1: LINEAR REGRESSION 
print("\n1. Training Linear Regression...")
lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print("   -> Done.")

#  MODEL 2: RANDOM FOREST 
print("\n2. Training Random Forest (this may take 1-2 minutes)...")
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("   -> Done.")

# --- EVALUATION FUNCTION (Now with MSE) ---
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred) # <--- Added MSE here
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    
   
    return [name, mae, mse, rmse, r2]

# Create the Results Table
results_df = pd.DataFrame([
    evaluate_model("Linear Regression", y_test, lr_preds),
    evaluate_model("Random Forest", y_test, rf_preds)
], columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"])

print("\n--- FINAL COMPARISON RESULTS ---")
pd.options.display.float_format = '{:,.2f}'.format 
print(results_df)