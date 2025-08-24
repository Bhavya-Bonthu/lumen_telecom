import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import hashlib
from urllib.parse import quote_plus
from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import io
# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Telecom Inventory Management", layout="wide")

# ======= MONGO CONNECTION =======
def get_db():
    """
    Connect to MongoDB Atlas (preferred) with safe URL encoding;
    if it fails, fallback to local MongoDB.
    """
    username = "hemanthnalla234"     # <-- change if needed
    password = "Hem@nth5691"         # <-- change if needed
    escaped_pw = quote_plus(password)

    atlas_uri = f"mongodb+srv://{username}:{escaped_pw}@cluster0.jwqs2z8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    try:
        client = MongoClient(atlas_uri, tls=True, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=8000)
        client.server_info()  # force connection
        return client["TelecomInventoryManagement"]
    except Exception:
        # Fallback to local MongoDB
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        return client["TelecomInventoryManagement"]

db = get_db()
COL_USERS = db["Users"]
COL_PRODUCTS = db["Products"]
COL_SUPPLIERS = db["Suppliers"]
COL_ORDERS = db["Orders"]
COL_NOTIFICATIONS = db["Notifications"]
# ==============================
# UTILS
# ==============================
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def check_password(pw: str, hashed: str) -> bool:
    return hash_password(pw) == hashed

def df_from_cursor(cur):
    data = list(cur)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "_id" in df.columns:
        df.rename(columns={"_id": "ID"}, inplace=True)
        df["ID"] = df["ID"].astype(str)
    return df
def get_users_df():
    return df_from_cursor(COL_USERS.find({}))

def get_products_df():
    return df_from_cursor(COL_PRODUCTS.find({}))

def get_suppliers_df():
    return df_from_cursor(COL_SUPPLIERS.find({}))

def get_orders_df():
    return df_from_cursor(COL_ORDERS.find({}))

def get_notifications_df():
    return df_from_cursor(COL_NOTIFICATIONS.find({}).sort("Created_At", -1))

def ensure_indexes():
    COL_USERS.create_index("Username", unique=True)
    COL_PRODUCTS.create_index([("Name", 1), ("Category", 1)], unique=False)
    COL_SUPPLIERS.create_index("Name", unique=False)
    COL_ORDERS.create_index("Order_Date")
ensure_indexes()

def seed_minimum_admin():
    if COL_USERS.count_documents({}) == 0:
        COL_USERS.insert_one({
            "Username": "admin",
            "Password": hash_password("admin123"),
            "Role": "Admin"
        })
seed_minimum_admin()

def post_notification(product_id: str, message: str):
    COL_NOTIFICATIONS.insert_one({
        "Product_ID": product_id,
        "Message": message,
        "Created_At": dt.datetime.utcnow()
    })
