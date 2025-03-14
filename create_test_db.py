import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_test_database(db_path='sample.db'):
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY,
        username TEXT NOT NULL,
        email TEXT NOT NULL,
        signup_date DATE NOT NULL,
        last_login DATE
    )
    ''')
    
    # Create products table
    cursor.execute('''
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        price REAL NOT NULL,
        category TEXT NOT NULL,
        inventory_count INTEGER NOT NULL
    )
    ''')
    
    # Create orders table
    cursor.execute('''
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        order_date DATE NOT NULL,
        total_amount REAL NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')
    
    # Create order_items table
    cursor.execute('''
    CREATE TABLE order_items (
        item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        price_per_unit REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders (order_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    # Generate sample user data
    usernames = ['john_doe', 'jane_smith', 'bob_jones', 'alice_wonder', 'sam_fisher', 
                 'lisa_king', 'mark_taylor', 'emma_white', 'tom_brown', 'sarah_green']
    emails = [f'{name.split("_")[0]}.{name.split("_")[1]}@example.com' for name in usernames]
    
    today = datetime.now().date()
    signup_dates = [today - timedelta(days=np.random.randint(1, 365)) for _ in range(len(usernames))]
    last_logins = [date + timedelta(days=np.random.randint(1, 30)) for date in signup_dates]
    
    users_data = []
    for i in range(len(usernames)):
        users_data.append((i+1, usernames[i], emails[i], signup_dates[i], last_logins[i]))
    
    cursor.executemany('INSERT INTO users VALUES (?, ?, ?, ?, ?)', users_data)
    
    # Generate sample product data
    product_categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
    product_names = [
        'Smartphone', 'Laptop', 'Headphones', 'T-shirt', 'Jeans',
        'Dress', 'Novel', 'Cookbook', 'Chair', 'Desk',
        'Lamp', 'Basketball', 'Tennis Racket', 'Running Shoes', 'Water Bottle'
    ]
    product_descriptions = [f'A high-quality {name.lower()}' for name in product_names]
    prices = [np.random.uniform(10, 1000) for _ in range(len(product_names))]
    categories = [product_categories[np.random.randint(0, len(product_categories))] for _ in range(len(product_names))]
    inventory = [np.random.randint(0, 100) for _ in range(len(product_names))]
    
    products_data = []
    for i in range(len(product_names)):
        products_data.append((i+1, product_names[i], product_descriptions[i], prices[i], categories[i], inventory[i]))
    
    cursor.executemany('INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)', products_data)
    
    # Generate sample orders data
    num_orders = 50
    user_ids = [np.random.randint(1, len(usernames)+1) for _ in range(num_orders)]
    order_dates = [today - timedelta(days=np.random.randint(1, 90)) for _ in range(num_orders)]
    order_statuses = ['Completed', 'Processing', 'Shipped', 'Cancelled']
    statuses = [order_statuses[np.random.randint(0, len(order_statuses))] for _ in range(num_orders)]
    
    orders_data = []
    order_items_data = []
    item_id = 1
    
    for i in range(num_orders):
        # Generate between 1 and 5 items per order
        num_items = np.random.randint(1, 6)
        total_amount = 0
        
        for j in range(num_items):
            product_id = np.random.randint(1, len(product_names)+1)
            quantity = np.random.randint(1, 5)
            price = prices[product_id-1]
            
            order_items_data.append((
                item_id,
                i+1,  # order_id
                product_id,
                quantity,
                price
            ))
            
            total_amount += quantity * price
            item_id += 1
        
        orders_data.append((
            i+1,  # order_id
            user_ids[i],
            order_dates[i],
            total_amount,
            statuses[i]
        ))
    
    cursor.executemany('INSERT INTO orders VALUES (?, ?, ?, ?, ?)', orders_data)
    cursor.executemany('INSERT INTO order_items VALUES (?, ?, ?, ?, ?)', order_items_data)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"Test database created at {db_path}")
    print(f"Tables created: users, products, orders, order_items")
    print(f"Sample data: {len(usernames)} users, {len(product_names)} products, {num_orders} orders")

if __name__ == "__main__":
    create_test_database() 