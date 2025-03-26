import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as stats



# Extract data from the DataFrame
df_year = np.array([1917, 1918, 1920, 1921, 1922, 1924, 1927, 1928, 1928, 1928, 1929, 1930, 1930, 1932, 1935, 1935, 1936, 1937]).reshape(-1, 1)
df_oppose = np.array([29, 28, 34, 33, 36, 29, 35, 29, 25, 29, 22, 22, 29, 31, 19, 29, 15, 17])
df_support = np.array([11, 12, 5, 8, 7, 10, 9, 8, 12, 7, 8, 13, 12, 8, 14, 9, 17, 17])
df_pyear = np.array([1917, 1918, 1922, 1924, 1930, 1935]).reshape(-1, 1)


# Function to plot with whole-number axes
def plot_regression(x, y, xlabel, ylabel, title):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', label='Linear regression line')

    # Set whole number ticks for both axes
    plt.xticks(np.arange(min(x), max(x) + 1, 1), rotation=45)
    plt.yticks(np.arange(min(y), max(y) + 1, 1))

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Plot for Opposed Score
plot_regression(df_year, df_oppose, 'Year', 'Opposed Score', 'Linear Regression: Opposed Elements Score Over Time')

# Plot for Supported Score
plot_regression(df_year, df_support, 'Year', 'Supported Score', 'Linear Regression: Supported Elements Score Over Time')

# Linear regression with statsmodels for opposed score
X = stats.add_constant(df_year)  # Add constant for intercept term
model = stats.OLS(df_oppose, X)
results = model.fit()

# Print the regression equation
intercept = results.params[0]
slope = results.params[1]
print(f"Opposed Regression Equation: y = {slope:.2f}x + {intercept:.2f}")

# Print the p-values
print("\nP-values for Opposed:")
print(results.pvalues)

# Linear regression with statsmodels for supported score
model2 = stats.OLS(df_support, X)
results2 = model2.fit()

# Print the regression equation
intercept2 = results2.params[0]
slope2 = results2.params[1]
print(f"Supported Regression Equation: y = {slope2:.2f}x + {intercept2:.2f}")

# Print the p-values
print("\nP-values for Supported:")
print(results2.pvalues)
