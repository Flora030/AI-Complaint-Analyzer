import sqlite3

def init_db():
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint TEXT,
        summary TEXT,
        category TEXT,
        severity TEXT,
        sentiment TEXT,
        response TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_complaint(data):
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO complaints (complaint, summary, category, severity, sentiment, response)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data["complaint"],
        data["summary"],
        data["category"],
        data["severity"],
        data["sentiment"],
        data["response"]
    ))

    conn.commit()

    complaint_id = cursor.lastrowid
    conn.close()
    return complaint_id

def clear_complaints():
    try:
        conn = sqlite3.connect("complaints.db")
        cursor = conn.cursor()

        cursor.execute("DELETE FROM complaints")

        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        print(f"Error clearing complaints: {e}")

#clear_complaints()
