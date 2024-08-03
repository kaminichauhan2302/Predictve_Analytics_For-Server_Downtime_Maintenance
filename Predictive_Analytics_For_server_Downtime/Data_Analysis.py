import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the plot style
sns.set(style="white")

# Connect to the SQLite database
conn = sqlite3.connect(r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db')

# Fetch data
df = pd.read_sql_query("SELECT * cleaned_your_table", conn)

# Print the columns to verify the structure
print("Columns in the DataFrame:", df.columns)

# Close the connection
conn.close()

# Display the first few rows
print(df.head())

# Check data types and missing values
print(df.info())

# Display basic statistics
print(df.describe())

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Count missing values
print(df.isnull().sum())

# Distribution of server statuses
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Server_ID')
plt.title('Distribution of Server Statuses')
plt.show()

# Distribution of numerical features
numerical_features = ['time_since_last_error']

for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Boxplot of error_id by server status
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Downtime_Duration', y='Server_ID')
plt.title('Server ID by Downtime Duration')
plt.show()

# Pairplot of numerical features
sns.pairplot(df[numerical_features])
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr_matrix = df[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Set the timestamp as the index
df.set_index('Timestamp', inplace=True)

# Resample the data to daily frequency and plot
df_resampled = df.resample('D').mean()

plt.figure(figsize=(12, 6))
plt.plot(df_resampled.index, df_resampled['Server_ID'])
plt.title('Daily Average Error ID Over Time')
plt.xlabel('Date')
plt.ylabel('Average Server ID')
plt.show()
