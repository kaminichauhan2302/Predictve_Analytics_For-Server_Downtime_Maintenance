import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Connect to the SQLite database
conn = sqlite3.connect(r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db')

# Fetch data
df = pd.read_sql_query("SELECT * FROM cleaned_your_table_name", conn)

# Data cleaning
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp')
df.fillna(0, inplace=True)

# Feature engineering
df['time_since_last_error'] = df.groupby('Server_ID')['Timestamp'].diff().fillna(pd.Timedelta(seconds=60)).astype('timedelta64[s]')
#df['rolling_avg_error_code'] = df.groupby('Server_ID')['error_code'].rolling(window=3).mean().reset_index(level=0, drop=True)
#df['error_count_24h'] = df.groupby('Server_ID')['error_code'].rolling(window='1D', on='Timestamp').count().reset_index(level=0, drop=True)

# Label creation
df['Downtime_label'] = df['Downtime_Duration'].shift(-1).apply(lambda x: 1 if x == 'DOWN' else 0)

# Train-test split
features = ['time_since_last_error',]
X = df[features]
y = df['Downtime_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save transformed data
df.to_sql('transformed_server_logs', conn, if_exists='replace', index=False)

# Close the connection
conn.commit()
conn.close()
