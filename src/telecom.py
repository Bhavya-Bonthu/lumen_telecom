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
