import pandas as pd
import sqlite3

conn = sqlite3.connect('complaints.db')
# df = pd.read_sql_query("SELECT * FROM complaints", conn)
# print(df)
# conn.close()

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM complaint_embeddings")
print(cur.fetchone())
conn.close()