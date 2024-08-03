import sqlite3
import pandas as pd

def handle_missing_values(df, strategy='drop', fill_value=None):
    
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill' and fill_value is not None:
        return df.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy or fill_value not provided for 'fill' strategy")

# Connect to the source databases
conn1 = sqlite3.connect(r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db')



# Create a cursor for each connection
cursor1 = conn1.cursor()




# Extract data from db
your_table_name_df = pd.read_sql_query('SELECT * FROM your_table_name', conn1)

# Handle missing values in the customers data
your_table_name_df = handle_missing_values(your_table_name_df, strategy='fill', fill_value={'Server_ID': 'None', 'Timestamp': 'None','CPU_Usage': 'None' ,'Memory-usage':'None' ,'Disk_Usage':'none', 'Network_Usage':'None', 'Downtime_Duration':'None', 'Maintenance_Type':'None' ,'Maintanance_Cost':'None'})



# Commit the changes and close all connections


conn1.close()


print("Data integration and missing value handling complete.")
