import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import linprog

# Step 1: Data Preparation
conn = sqlite3.connect(r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db')
query = """
SELECT Server_ID, Timestamp, Downtime_Duration, Maintenance_Type
FROM cleaned_your_table_name
"""
data = pd.read_sql_query(query, conn)
conn.close()

# Convert MaintenanceDate to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

# Handle cases where the conversion failed
data = data.dropna(subset=['Timestamp'])

# Step 2: Feature Engineering
data['Year'] = data['Timestamp'].dt.year
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek

# Step 3: EDA
plt.figure(figsize=(12, 6))
sns.lineplot(x='Timestamp', y='Downtime_Duration', data=data)
plt.title('Downtime Duration Over Time')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Server_ID', y='Downtime_Duration', data=data)
plt.title('Downtime Duration by Server')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='Downtime_Duration', data=data)
plt.title('Downtime Duration by Month')
plt.show()

# Step 4: Predictive Modeling
X = data[['Server_ID', 'Year', 'Month', 'DayOfWeek']]
y = data['Downtime_Duration']
X = pd.get_dummies(X, columns=['Server_ID'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))

# Step 5: Optimization
c = data['Downtime_Duration'].values

# Define constraints for linear programming example (these need to be defined based on your requirements)
# For example, we could assume that each server needs to be maintained once per period (e.g., day)
num_periods = len(data['Timestamp'].unique())
num_servers = len(data['Server_ID'].unique())
A = [] # Placeholder for constraints matrix (size should be num_periods x num_servers)
b = []  # Placeholder for constraints vector (length should be num_periods)

# For the sake of demonstration, let's assume a simple example where each server can only be maintained once per period
# A = [[1, 0, ..., 0],
#      [0, 1, ..., 0],
#      ...
#      [0, 0, ..., 1]]
# b = [1, 1, ..., 1]
# Define objective function
c = y.values

# Define constraints
num_vars = len(c)  # Number of variables
num_constraints = len(data['Timestamp'].unique())  # Example constraint count

# Initialize constraints matrix and vector
A_eq = []
b_eq = []

# Example constraints: Each server must be maintained exactly once per period
for period in range(num_constraints):
    constraint_row = [0] * num_vars
    for i, column in enumerate(X.columns):
        if column.startswith('Server_ID'):
            constraint_row[i] = 1
    A_eq.append(constraint_row)
    b_eq.append(1)  # Example: One maintenance per period

# Convert to numpy arrays for linprog
import numpy as np
A_eq = np.array(A_eq)
b_eq = np.array(b_eq)
bounds = [(0, 1)] * num_vars
# Bounds for decision variables
x_bounds = (0, 1)

res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
print('Optimal Maintenance Schedule:', res.x)
print('Total Downtime with Optimized Schedule:', res.fun)

# Step 6: Evaluation
optimized_schedule = res.x
total_downtime = sum(optimized_schedule * data['Downtime_Duration'].values)
print('Total Downtime with Optimized Schedule:', total_downtime)
