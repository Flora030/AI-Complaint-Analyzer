import pandas as pd

df = pd.read_sql_query("SELECT * FROM complaints", 'sqlite:///complaints.db')
print(df)