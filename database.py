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
    conn.close()


def get_all_complaints():
    conn = sqlite3.connect("complaints.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM complaints")
    rows = cursor.fetchall()

    conn.close()
    return rows
