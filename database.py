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
    #print("Saving complaint to DB:", data)
    response_to_save = data.get("response", "No response provided.")
    
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
        response_to_save
    ))

    conn.commit()
    conn.close()

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
