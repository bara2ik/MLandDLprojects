import pandas as pd
import glob
import os


# LOAD AND COMBINE ALL DATASETS 
# Get a list of all CSV files in the current directory (excluding cleaned_car_data.csv)
file_list = [f for f in glob.glob("*.csv") if f != "cleaned_car_data.csv"]

# List to hold dataframes
dfs = []

print("Loading files...")
for file in file_list:
    # Read the file
    df = pd.read_csv(file)
    
    # Create a 'brand' column based on the filename (e.g., 'audi.csv' -> 'audi')
    brand_name = os.path.basename(file).replace(".csv", "")
    df['brand'] = brand_name
    
    dfs.append(df)
    print(f" -> Loaded {file} with {len(df)} rows.")

# Combine them into one big dataframe
full_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal Raw Data: {len(full_df)} rows")

#  CLEANING INCONSISTENT COLUMNS 
# Some files have 'tax', others have 'tax(£)'. We merge them.
if 'tax(£)' in full_df.columns:
    full_df['tax'] = full_df['tax'].fillna(full_df['tax(£)'])
    full_df = full_df.drop(columns=['tax(£)'])
    print(" -> Merged 'tax(£)' into 'tax' column.")

# REMOVING DUPLICATES 
initial_count = len(full_df)
full_df = full_df.drop_duplicates()
print(f" -> Removed {initial_count - len(full_df)} duplicate rows.")

# HANDLING MISSING VALUES 
full_df['tax'] = full_df['tax'].fillna(full_df['tax'].median())
full_df['mpg'] = full_df['mpg'].fillna(full_df['mpg'].median())
print(" -> Filled missing values in 'tax' and 'mpg' with median.")

#  FEATURE ENGINEERING
current_year = 2025
full_df['car_age'] = current_year - full_df['year']


full_df = full_df.drop(columns=['year'])
print(" -> Created 'car_age' column and removed 'year'.")




#  SAVE AND PREVIEW 
full_df.to_csv("cleaned_car_data.csv", index=False)

print("\nSUCCESS! Preprocessing Complete.")
print(f"Final Dataset Size: {len(full_df)} rows")
print("\nFirst 5 rows of your clean data:")
print(full_df.head())