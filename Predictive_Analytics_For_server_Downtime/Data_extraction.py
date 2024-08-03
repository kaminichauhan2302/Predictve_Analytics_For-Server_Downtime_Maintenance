import sqlite3
# The database file name with Unicode characters
database_file = r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db' 
#database file name

# Encode the file name as UTF-8
database_file_encoded = database_file.encode('utf-8')

# Decode the file name to ensure it's handled correctly by sqlite3
database_file_decoded = database_file_encoded.decode('utf-8')

# Connect to the SQLite database
conn = sqlite3.connect(database_file_decoded)

# Create a cursor object
cursor = conn.cursor()

# Execute a query
cursor.execute('SELECT * FROM your_table_name')  

# Fetch the data
rows = cursor.fetchall()

# Optional: Fetch column names
column_names = [description[0] for description in cursor.description]

# Print the data
print("Column Names:", column_names)
for row in rows:
    print(row)

# Close the connection
conn.close()
