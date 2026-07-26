import sqlite3
from config.settings import DB_PATH

def get_connection():
    return sqlite3.connect(database=DB_PATH, check_same_thread=False)

def init_db():
    """Initializes the required database schema."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_conversation(thread_id: str, title: str = "New Conversation"):
    """Inserts a new conversation thread."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO conversations (thread_id, title)
            VALUES (?, ?)
        """, (str(thread_id), title))
        conn.commit()

def update_title(thread_id: str, title: str):
    """Updates the conversation title."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE conversations
            SET title = ?
            WHERE thread_id = ?
        """, (title, str(thread_id)))
        conn.commit()

def delete_conversation(thread_id: str):
    """Deletes a conversation thread by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM conversations WHERE thread_id = ?
        """, (str(thread_id),))
        conn.commit()

def retrieve_all_threads() -> list[str]:
    """Retrieves all thread IDs ordered by newest first."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT thread_id FROM conversations ORDER BY rowid DESC
        """)
        return [row[0] for row in cursor.fetchall()]

def retrieve_all_titles() -> dict[str, str]:
    """Retrieves a dictionary mapping thread_id to title."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT thread_id, title FROM conversations ORDER BY rowid DESC
        """)
        return {row[0]: row[1] for row in cursor.fetchall()}