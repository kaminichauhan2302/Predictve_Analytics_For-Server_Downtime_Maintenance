import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Data Extraction
conn = sqlite3.connect(r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db')
query = "SELECT * FROM transformed_server_logs"
df = pd.read_sql_query(query, conn)
conn.close()

# Check initial DataFrame
print("Initial DataFrame:\n", df.head())
print("Number of rows in DataFrame:", len(df))

# Ensure 'timestamp' is in datetime format
df['timestamp'] = pd.to_datetime(df['Timestamp'])

# Step 2: Data Preprocessing
print("Number of rows before preprocessing:", len(df))
df = df.dropna()
print("Number of rows after dropping NA:", len(df))

# Create lag features and rolling statistics
df['cpu_lag1'] = df['CPU_Usage'].shift(1)
df['mem_lag1'] = df['Memory_Usage'].shift(1)
df['cpu_roll_mean'] = df['CPU_Usage'].rolling(window=24).mean()
df['mem_roll_std'] = df['Memory_Usage'].rolling(window=24).std()
df.dropna(inplace=True)  # Drop rows with NaN values

print("DataFrame after feature engineering:\n", df.head())
print("Number of rows after feature engineering:", len(df))

# Feature Engineering
df['previous_downtime'] = df['Downtime_Duration'].shift(1)
df['previous_downtime'].fillna(0, inplace=True)
df['cpu_mem_ratio'] = df['CPU_Usage'] / df['Memory_Usage']

# Verify features and target columns
features = ['cpu_lag1', 'mem_lag1', 'cpu_roll_mean', 'mem_roll_std', 'previous_downtime', 'cpu_mem_ratio']
target = 'Downtime_Duration'

print("Features used for training:", features)
print("Target column:", target)

for col in features + [target]:
    if col not in df.columns:
        raise ValueError(f"Column {col} is missing from the DataFrame")

# Ensure there are enough rows to split
if len(df) > 1:
    test_size = 0.1  # Adjust test_size based on the size of your dataset
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=test_size, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
else:
    # Use cross-validation for small datasets
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, df[features], df[target], cv=5)
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())

# Step 5: Model Deployment
joblib.dump(model, 'server_downtime_predictor.pkl')
loaded_model = joblib.load('server_downtime_predictor.pkl')
example_data = df[features].iloc[0]  # Example data from the dataset
prediction = loaded_model.predict([example_data])
print("Predicted Downtime Occurrence:", prediction)
