import sqlite3

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row  # To access columns by name
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create a users table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        remember_me BOOLEAN DEFAULT 0
    )''')
    conn.commit()
    conn.close()

