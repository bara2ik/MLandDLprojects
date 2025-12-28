import pandas as pd
import joblib



model = joblib.load('car_price_model.pkl')
   


# DEFINE YOUR NEW CAR
new_car_data = {
    'brand': ['Audi'],           # Options: Audi, BMW, Ford, VW, etc.
    'transmission': ['Automatic'],  # Options: Manual, Automatic, Semi-Auto
    'fuelType': ['Diesel'],      # Options: Petrol, Diesel, Hybrid
    'mileage': [25952],          # Miles driven
    'tax': [145],                # Road tax
    'mpg': [67.3],               # Miles Per Gallon
    'engineSize': [2.0],         # Engine size in Liters
    'car_age': [8]               # Age of car (2025 - Year)
}

# Convert dictionary to a pandas DataFrame
new_car_df = pd.DataFrame(new_car_data)

print("\n--- PREDICTING PRICE FOR NEW ENTRY ---")
print(new_car_df)

# MAKE PREDICTION
predicted_price = model.predict(new_car_df)
print(f"\nEstimated Market Value: £{predicted_price[0]:,.2f}")