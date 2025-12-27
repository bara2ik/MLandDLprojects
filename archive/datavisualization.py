import matplotlib


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Try to use an interactive backend for displaying plots
try:
    matplotlib.use('TkAgg')  # Windows compatible
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass  # Use default backend

# Disable interactive mode to ensure plots stay open (block=True will keep them open)
plt.ioff()

# 1. Load the clean data
os.path.exists("cleaned_car_data.csv"):
df = pd.read_csv("cleaned_car_data.csv")
print("Loaded cleaned_car_data.csv")
print(f"Dataset shape: {df.shape}")


# Set the style for professional-looking charts
sns.set_style("whitegrid")

# --- CHART 1: Correlation Matrix (Rubric Requirement) ---
plt.figure(figsize=(10, 6))
# Select only numbers for correlation
numerical_data = df[['price', 'mileage', 'tax', 'mpg', 'engineSize', 'car_age']]
correlation = numerical_data.corr()
# Draw the heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix: What features affect Price?", fontsize=14)
plt.tight_layout()
plt.show(block=True)

# --- CHART 2: Scatter Plot (Price vs. Mileage) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='mileage', y='price', data=df, alpha=0.3, color='blue')
plt.title("Price vs. Mileage (Validation of Logic)", fontsize=14)
plt.xlabel("Mileage (Miles)")
plt.ylabel("Price (£)")
# Add a trendline to look fancy
sns.regplot(x='mileage', y='price', data=df, scatter=False, color='red')
plt.tight_layout()
plt.show(block=True)

# --- CHART 3: Box Plot (Price by Transmission) ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='transmission', y='price', data=df, palette="Set2")
plt.title("Price Distribution by Transmission Type", fontsize=14)
plt.tight_layout()
plt.show(block=True)

# Keep the script running to view plots
input("\nPress Enter to close all plots and exit...")