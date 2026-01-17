import sqlite3
import hashlib

def create_user_table():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email, password, role="user"):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (email, password, role) VALUES (?, ?, ?)",
            (email, hash_password(password), role)
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        "SELECT role FROM users WHERE email=? AND password=?",
        (email, hash_password(password))
    )
    result = c.fetchone()
    conn.close()
    return result
