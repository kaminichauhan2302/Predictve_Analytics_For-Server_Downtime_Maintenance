import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and Prepare Data
conn = sqlite3.connect(r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db')
query = """
SELECT Server_ID, Timestamp, Downtime_Duration, Maintenance_Type
FROM cleaned_your_table_name
"""
data = pd.read_sql_query(query, conn)
conn.close()

# Convert MaintenanceDate to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
data = data.dropna(subset=['Timestamp'])

# Extract additional features
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek

# One-hot encoding for categorical features
data_encoded = pd.get_dummies(data, columns=['Server_ID', 'Maintenance_Type'])

# Step 2: Predictive Maintenance Modeling
# Define target variable (binary classification: downtime > threshold)
threshold = data['Downtime_Duration'].median()
data_encoded['Failure'] = (data['Downtime_Duration'] > threshold).astype(int)

# Define features and target
columns_to_drop = ['Downtime_Duration', 'Timestamp', 'Failure']
X = data_encoded.drop(columns=columns_to_drop)
y = data_encoded['Failure']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model performance
print("Predictive Maintenance Model Performance:")
print(classification_report(y_test, y_pred))

# Step 3: Optimize Maintenance Scheduling
# Define the objective function (minimize downtime)
c = data['Downtime_Duration'].values

# Example constraints
num_tasks = len(c)
num_periods = len(data['Timestamp'].unique())
A_eq = np.ones((num_periods, num_tasks))  # Example: total maintenance tasks per period
b_eq = np.array([num_tasks // num_periods] * num_periods)  # Example: ensure tasks are distributed

# Bounds for each decision variable (0 or 1)
bounds = [(0, 1)] * num_tasks

# Perform optimization
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

print("Optimal Maintenance Schedule:")
print(result.x)
print("Total Downtime with Optimized Schedule:")
print(result.fun)

# Step 4: Visualize and Analyze Results
# Downtime Duration Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Timestamp', y='Downtime_Duration', data=data)
plt.title('Downtime Duration Over Time')
plt.xlabel('Date')
plt.ylabel('Downtime Duration')
plt.show()

# Downtime Duration by Server
plt.figure(figsize=(12, 6))
sns.barplot(x='Server_ID', y='Downtime_Duration', data=data)
plt.title('Downtime Duration by Server')
plt.xticks(rotation=90)
plt.show()

# Downtime Duration by Month
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Downtime_Duration', data=data)
plt.title('Downtime Duration by Month')
plt.show()
