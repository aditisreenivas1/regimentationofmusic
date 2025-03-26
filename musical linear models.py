import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

# Load CSV
df = pd.read_csv("/Users/aditisreenivas/Downloads/classical trends.csv", header=None)

# Convert relevant columns to numeric (assuming they contain numbers)
for col in range(2, 12):  # Columns 2 to 11 (Year + Metrics)
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert and handle errors

# Transpose and fix column headers
df = df.T.reset_index(drop=True)  # Reset index so columns remain numeric
df.columns = range(df.shape[1])  # Ensure column names are integer-based

# Convert all columns to numeric again after transpose
df = df.apply(pd.to_numeric, errors='coerce')

dataprok = []
for i in df:
    if df.iloc[i,3] == "Prokofiev":
    dataprok[i] = [df[i,2],df[i,3], df[i,4],df[i,6],df[i,7],df[i,8],df[i,9],df[i,10]]



# Define data dictionary with corrected keys
data = {
    "Dynamic Contrast": {
        "Year": df[2].values,
        "Value": df[5].values,
        "color": "blue"
    },
    "Interval Complexity": {
        "Year": df[2].values,
        "Value": df[6].values,
        "color": "green"
    },
    "Syncopation": {
        "Year": df[2].values,
        "Value": df[7].values,
        "color": "red"
    },
    "Sixteenth Notes": {
        "Year": df[2].values,
        "Value": df[8].values,
        "color": "yellow"
    },
    "Polyphony": {
        "Year": df[2].values,
        "Value": df[9].values,
        "color": "orange"
    },
    "Straight Rhythms": {
        "Year": df[2].values,
        "Value": df[10].values,
        "color": "purple"
    },
    "Homophony": {
        "Year": df[2].values,
        "Value": df[11].values,
        "color": "brown"
    }
}

# Plot setup
plt.figure(figsize=(10, 6))

for group, values in data.items():
    x, y, color = values["Year"], values["Value"], values["color"]

    # Remove NaN values before regression
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    if len(x) < 2:  # Ensure there are enough points for regression
        print(f"Skipping {group} due to insufficient data.")
        continue

    # Perform linear regression
    slope, intercept, r_value, _, _ = linregress(x, y)
    line = slope * x + intercept  # Trend line

    # Scatter plot
    plt.scatter(x, y, color=color, label=f'{group} Data')

    # Trend line
    plt.plot(x, line, color=color, linestyle="--", label=f'{group} Trend')

    # Show equation on plot
    eq_text = f'{group}: y = {slope:.2f}x + {intercept:.2f}, R² = {r_value**2:.2f}'
    plt.text(x[-1], line[-1], eq_text, fontsize=9, color=color)

# Labels and title
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Multiple Groups Scatter Plot with Trend Lines')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
