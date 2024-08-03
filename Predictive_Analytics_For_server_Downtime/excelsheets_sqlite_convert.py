import pandas as pd
import sqlite3
import os
import openpyxl



# Load the Excel file
excel_file_path = r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\Server Data (1).xlsx'
df = pd.read_excel(excel_file_path)  # Adjust sheet_name as needed

try:
    if os.path.exists(excel_file_path):

        os.chmod(excel_file_path, 0o666)
        print("File permissions modified successfully!")
    else:
        print("File not found:", excel_file_path)
except PermissionError:
    print("Permission denied: You don't have the necessary permissions to change the permissions of this file.")


# Connect to SQLite database (it will create the database file if it doesn't exist)
sqlite_file = 'your_database.db'
conn = sqlite3.connect(sqlite_file)

# Write the data to SQLite (replace 'your_table_name' with the actual table name)
df.to_sql('your_table_name', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Data has been saved to SQLite database.")
