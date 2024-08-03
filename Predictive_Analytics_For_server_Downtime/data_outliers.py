import sqlite3
import pandas as pd

def remove_outliers_iqr(df, column_name):
    """
    Remove outliers from a DataFrame using the IQR method.
    
    Parameters:
    - df: DataFrame
    - column_name: Name of the column from which to remove outliers
    
    Returns:
    - DataFrame with outliers removed
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    
    # Ensure the column is numeric
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    # Drop rows where conversion to numeric fails
    df = df.dropna(subset=[column_name])
    
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

def clean_server_metrics(database_file, table_name, columns_to_clean):
    # Connect to the database
    conn = sqlite3.connect(database_file)
    
    # Load data into a DataFrame
    df = pd.read_sql_query(f'SELECT * FROM {table_name}', conn)
    
    # Verify columns and remove outliers
    for column in columns_to_clean:
        if column in df.columns:
            df = remove_outliers_iqr(df, column)
        else:
            print(f"Warning: Column '{column}' not found in '{table_name}' table.")
    
    # Save the cleaned data back to a new table in the database
    cleaned_table_name = f'cleaned_{table_name}'
    df.to_sql(cleaned_table_name, conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Outliers removed from {table_name} and cleaned data saved to {cleaned_table_name}.")

# Specify the database file, table name, and columns to clean
database_file = r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db'
table_name = 'your_table_name'
columns_to_clean = ['Downtime_Duration', 'Maintenance_Cost', 'Server_ID','Memory_Usage','CPU_Usage','Network_Usage', 'Disk_Usage']

# Use the function to clean server metrics data
clean_server_metrics(database_file, table_name, columns_to_clean)
