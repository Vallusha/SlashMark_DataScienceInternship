import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load the Data
df = pd.read_csv('weather.csv')

# Step 2: Data Exploration
print("First few rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(df['Rainfall'], bins=20, kde=True)
plt.xlabel('Rainfall')
plt.ylabel('Frequency')
plt.title('Distribution of Rainfall')
plt.grid(True)
plt.show()

# Step 4: Feature Engineering (if needed)
# Example: Create a new feature 'Season' based on the month
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'].apply(lambda month: 'Spring' if month in [3, 4, 5] else ('Summer' if month in [6, 7, 8] else ('Fall' if month in [9, 10, 11] else 'Winter')))

# Step 5: Data Analysis (analyze each term)
# Example: Calculate average MaxTemp and Rainfall by Season
seasonal_avg_max_temp = df.groupby('Season')['MaxTemp'].mean()
seasonal_avg_rainfall = df.groupby('Season')['Rainfall'].mean()

# Step 6: Data Visualization (Part 2)
plt.figure(figsize=(10, 5))
sns.barplot(x=seasonal_avg_max_temp.index, y=seasonal_avg_max_temp.values)
plt.xlabel('Season')
plt.ylabel('Average Max Temperature')
plt.title('Seasonal Average Max Temperature')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=seasonal_avg_rainfall.index, y=seasonal_avg_rainfall.values)
plt.xlabel('Season')
plt.ylabel('Average Rainfall')
plt.title('Seasonal Average Rainfall')
plt.grid(True)
plt.show()

# Step 7: Advanced Analysis (e.g., predict Rainfall)
# Prepare the data for prediction
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'\nMean Squared Error for Rainfall Prediction: {mse:.2f}')
print(f'Mean Absolute Error for Rainfall Prediction: {mae:.2f}')
print(f'R-squared for Rainfall Prediction: {r2:.2f}')

# Step 8: Conclusions and Insights (analyze each term)
# Example: Identify the highest and lowest rainfall seasons
highest_rainfall_season = seasonal_avg_rainfall.idxmax()
lowest_rainfall_season = seasonal_avg_rainfall.idxmin()
print(f'\nHighest rainfall season: {highest_rainfall_season}, Lowest rainfall season: {lowest_rainfall_season}')
