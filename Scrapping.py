import requests
from lxml import html
import time
import mysql.connector

URL = "https://www.flipkart.com"
SEARCH_URL = "https://www.flipkart.com/search?q=watch&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

limit = 20  
product_links = []

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="csedept"
)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS watches (
        Brand VARCHAR(255),
        Name TEXT,
        Price FLOAT,
        Discount VARCHAR(50), 
        Rating FLOAT,
        Ratings_Count INT,
        Reviews_Count INT,
        Delivery VARCHAR(100)
    )
""")
cursor.execute("DELETE FROM watches")

def extract_text_xpath(tree, xpath_expr, default=""):
    try:
        return tree.xpath(xpath_expr)[0].strip()
    except IndexError:
        return default

def extract_float(text):
    try:
        return float(text.replace("₹", "").replace(",", "").strip())
    except:
        return 0.0

print("Collecting product links...")
page = 1
url = SEARCH_URL
while page <= limit:
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to retrieve page: {page}")
        break

    tree = html.fromstring(response.content)
    products = tree.xpath('//a[contains(@class, "WKTcLC")]/@href')

    for href in products:
        if href:
            product_links.append(URL + href.strip())

    next_page_links = tree.xpath('//a[contains(@class,"cn++Ap")]/@href')
    if next_page_links:
        url = URL + next_page_links[0]
        page += 1
        time.sleep(1)
    else:
        break

print(f"Collected {len(product_links)} product links.")
print("Scraping product details...")

for index, product_link in enumerate(product_links):
    response = requests.get(product_link, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to load: {product_link}")
        continue

    tree = html.fromstring(response.content)

    brand = extract_text_xpath(tree, '//span[contains(@class, "mEh187")]/text()', "Unknown Brand")
    name = extract_text_xpath(tree, '//span[contains(@class, "VU-ZEz")]/text()', "No Name")

    price_text = extract_text_xpath(tree, '//div[contains(@class,"Nx9bqj") or contains(@class,"_30jeq3")]/text()', "₹0")
    price = extract_float(price_text)

    discount = extract_text_xpath(tree, '//div[contains(@class, "UkUFwK")]/span/text()', "No Discount")
    
    rating_text = extract_text_xpath(tree, '//div[contains(@class, "XQDdHH")]/text()', "0")
    rating = extract_float(rating_text)

    delivery = extract_text_xpath(tree, '//span[contains(@class, "hcf08j")]/text()', "Not Free")

    try:
        rr_text = extract_text_xpath(tree, '//span[contains(@class, "Wphh3N")]/span/text()')
        parts = rr_text.split(" and ")
        ratings = int(parts[0].replace(",", "").split()[0]) if len(parts) > 0 else 0
        reviews = int(parts[1].replace(",", "").split()[0]) if len(parts) > 1 else 0
    except:
        ratings, reviews = 0, 0

    values = (brand, name, price, discount, rating, ratings, reviews, delivery)

    cursor.execute("""
        INSERT INTO watches (Brand, Name, Price, Discount, Rating, Ratings_Count, Reviews_Count, Delivery)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, values)
    conn.commit()

    print(f"Scraped {index + 1}/{len(product_links)}")
    time.sleep(1)

print("Scraping complete and data inserted into MySQL.")
cursor.close()
conn.close()
