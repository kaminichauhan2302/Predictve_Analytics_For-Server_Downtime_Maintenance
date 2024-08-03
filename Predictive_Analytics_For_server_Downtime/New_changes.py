from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_data():
    conn = sqlite3.connect(r'C:\Users\Dell\OneDrive\Desktop\Predictive_Analytics\your_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cleaned_your_table_name")
    rows = cursor.fetchall()
    conn.close()
    
    # Convert to JSON format
    result = [dict(zip([column[0] for column in cursor.description], row)) for row in rows]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
