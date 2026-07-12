import sqlite3
import bcrypt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "users.db"

def init_db():
    """
    Automatically creates the database and users table if they do not exist.
    This runs silently in the background when main.py starts.
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def authenticate_or_register(user_id: str, password: str) -> str:
    """
    Checks if user exists. 
    If yes: verifies password. Returns 'authenticated' or 'failed'.
    If no: hashes password, creates user. Returns 'registered'.
    """
    init_db()
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    
    if result:
        # User exists: Verify password
        stored_hash = result[0]
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            conn.close()
            return "authenticated"
        else:
            conn.close()
            return "failed"
    else:
        # User does not exist: Register them automatically
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        cursor.execute("INSERT INTO users (user_id, password_hash) VALUES (?, ?)", 
                       (user_id, hashed.decode('utf-8')))
        conn.commit()
        conn.close()
        return "registered"