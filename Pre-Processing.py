import pandas as pd
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="csedept"
)

query = "SELECT * FROM watches"
df = pd.read_sql(query, conn)


df["Name"] = df["Name"].astype(str).str.strip().str.lower()
df["Brand"] = df["Brand"].astype(str).str.strip().str.lower()

df["Price"] = df["Price"].astype(str).str.replace("₹", "").str.replace(",", "")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(float)

df["Discount"] = df["Discount"].astype(str).str.lower().str.strip()

df["Discount"] = df["Discount"].replace(
    ["no discount", "none", "nan", "no discounts", "no offer", "", "n/a", "no discount available"],"0"
)

df["Discount"] = df["Discount"].str.extract(r'(\d+)', expand=False)

df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0).astype(int)

df["Ratings_Count"] = df["Ratings_Count"].astype(str).str.replace(",", "").replace("No Ratings", "0")
df["Ratings_Count"] = pd.to_numeric(df["Ratings_Count"], errors="coerce").fillna(0).astype(int)

df["Reviews_Count"] = df["Reviews_Count"].astype(str).str.replace(",", "").replace("No Reviews", "0")
df["Reviews_Count"] = pd.to_numeric(df["Reviews_Count"], errors="coerce").fillna(0).astype(int)

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0).astype(float)

df = df.drop_duplicates(subset=["Name", "Brand", "Price"])

print(df.head())

cursor = conn.cursor()


for index, row in df.iterrows():
   
    update_query = """
        UPDATE watches
        SET Price = %s, Discount = %s, Rating = %s, Ratings_Count = %s, 
            Reviews_Count = %s
        WHERE Name = %s AND Brand = %s
    """
    cursor.execute(update_query, (
        row["Price"], row["Discount"], row["Rating"],
        row["Ratings_Count"], row["Reviews_Count"],
        row["Name"], row["Brand"]
    ))


conn.commit()
cursor.close()
conn.close()
