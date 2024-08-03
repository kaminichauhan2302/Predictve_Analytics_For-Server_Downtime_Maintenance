from flask import Flask, jsonify
import sqlite3
import pandas as pd

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('your_database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/server_list', methods=['GET'])
def get_users():
    conn = get_db_connection()
    server_list = conn.execute('SELECT * FROM transformed_server_logs').fetchall()
    conn.close()
    
    user_list = [dict(row) for row in server_list]
    return jsonify(user_list)

if __name__ == '__main__':
    app.run(debug=True)
