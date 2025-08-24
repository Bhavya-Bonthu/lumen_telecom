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
# ==============================
# AUTH
# ==============================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

def do_login(username, password):
    user = COL_USERS.find_one({"Username": username})
    if user and check_password(password, user["Password"]):
        # convert _id
        user["ID"] = str(user["_id"])
        st.session_state.auth_user = user
        return True
    return False

def do_signup(username, password, role):
    if COL_USERS.find_one({"Username": username}):
        st.error("Username already exists.")
        return
    COL_USERS.insert_one({
        "Username": username,
        "Password": hash_password(password),
        "Role": role
    })
    st.success("User created. Please Login.")

# ==============================
# CRUD HELPERS
# ==============================
def add_supplier(name, contact, rating, perf):
    COL_SUPPLIERS.insert_one({
        "Name": name,
        "Contact": contact,
        "Rating": rating,
        "Performance_Metrics": perf
    })

def update_supplier(supplier_id, name, contact, rating, perf):
    COL_SUPPLIERS.update_one(
        {"_id": ObjectId(supplier_id)},
        {"$set": {"Name": name, "Contact": contact, "Rating": rating, "Performance_Metrics": perf}}
    )

def delete_supplier(supplier_id):
    COL_SUPPLIERS.delete_one({"_id": ObjectId(supplier_id)})

def add_product(name, category, stock, supplier_id, reorder):
    COL_PRODUCTS.insert_one({
        "Name": name,
        "Category": category,
        "Stock": int(stock),
        "Supplier_ID": supplier_id,
        "Reorder_Level": int(reorder)
    })
    # alert if already low
    if int(stock) <= int(reorder):
        post_notification("", f"Low initial stock for {name}: {stock} <= reorder level {reorder}")

def update_product(prod_id, name, category, stock, supplier_id, reorder):
    COL_PRODUCTS.update_one(
        {"_id": ObjectId(prod_id)},
        {"$set": {"Name": name, "Category": category, "Stock": int(stock),
                  "Supplier_ID": supplier_id, "Reorder_Level": int(reorder)}}
    )

def delete_product(prod_id):
    COL_PRODUCTS.delete_one({"_id": ObjectId(prod_id)})

def place_order(product_id: str, qty: int, supplier_id: str, unit_price: float = 0.0):
    prod = COL_PRODUCTS.find_one({"_id": ObjectId(product_id)})
    if not prod:
        raise ValueError("Product not found.")
    current_stock = int(prod.get("Stock", 0))
    qty = int(qty)

    if current_stock < qty:
        raise ValueError("❌ Not enough stock available")

    # Insert order
    order_doc = {
        "Product_ID": product_id,
        "Supplier_ID": supplier_id,
        "Qty": qty,
        "Unit_Price": float(unit_price),
        "Order_Date": dt.datetime.utcnow()
    }
    COL_ORDERS.insert_one(order_doc)

    # Decrement stock
    COL_PRODUCTS.update_one(
        {"_id": ObjectId(product_id)},
        {"$inc": {"Stock": -qty}}
    )

    # check for low stock after order
    updated = COL_PRODUCTS.find_one({"_id": ObjectId(product_id)})
    if updated:
        if int(updated.get("Stock", 0)) <= int(updated.get("Reorder_Level", 0)):
            post_notification(product_id, f"⚠ Low stock for {updated['Name']}: {updated['Stock']} (≤ reorder {updated['Reorder_Level']})")

# ==============================
# EXPORT HELPERS
# ==============================
def download_df(df: pd.DataFrame, filename_base: str):
    c1, c2 = st.columns(2)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    c1.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"{filename_base}.csv", mime="text/csv")

    # Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    c2.download_button("⬇️ Download Excel", data=buffer.getvalue(), file_name=f"{filename_base}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==============================
# DASHBOARD CARDS
# ==============================
def dashboard_kpis():
    products = get_products_df()
    orders = get_orders_df()
    suppliers = get_suppliers_df()

    total_products = len(products)
    total_suppliers = len(suppliers)
    total_orders = len(orders)
    total_stock = int(products["Stock"].sum()) if not products.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products", total_products)
    c2.metric("Total Suppliers", total_suppliers)
    c3.metric("Total Orders", total_orders)
    c4.metric("Total Stock", total_stock)

def chart_inventory_levels():
    df = get_products_df()
    if df.empty:
        st.info("No products yet.")
        return
    fig = px.bar(df.sort_values("Stock", ascending=False), x="Name", y="Stock", color="Category", title="Inventory by Product")
    st.plotly_chart(fig, use_container_width=True)

def chart_category_split():
    df = get_products_df()
    if df.empty:
        return
    pie = df.groupby("Category", as_index=False)["Stock"].sum()
    fig = px.pie(pie, names="Category", values="Stock", title="Stock by Category")
    st.plotly_chart(fig, use_container_width=True)

def chart_monthly_orders():
    df = get_orders_df()
    if df.empty:
        st.info("No orders yet.")
        return
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
    m = df.groupby("YearMonth", as_index=False)["Qty"].sum()
    fig = px.line(m, x="YearMonth", y="Qty", title="Monthly Orders (Qty)")
    st.plotly_chart(fig, use_container_width=True)

def supplier_performance_table():
    orders = get_orders_df()
    products = get_products_df()
    suppliers = get_suppliers_df()
    if orders.empty or suppliers.empty or products.empty:
        st.info("Need suppliers, products and orders to compute performance.")
        return
    merged = orders.merge(products[["ID", "Name"]], left_on="Product_ID", right_on="ID", how="left")
    merged = merged.merge(suppliers[["ID", "Name"]].rename(columns={"Name": "Supplier_Name"}),
                          left_on="Supplier_ID", right_on="ID", how="left", suffixes=("", "_Sup"))
    perf = merged.groupby(["Supplier_ID", "Supplier_Name"], as_index=False).agg(
        Total_Orders=("Qty", "sum"),
        Distinct_Products=("Product_ID", "nunique")
    ).sort_values("Total_Orders", ascending=False)
    st.subheader("Supplier Performance")
    st.dataframe(perf, use_container_width=True)

# ==============================
# ML DEMAND FORECAST
# ==============================
def build_demand_features(order_df: pd.DataFrame):
    """Aggregate monthly demand per product and build simple time index features"""
    order_df = order_df.copy()
    if order_df.empty:
        return pd.DataFrame()
    order_df["Order_Date"] = pd.to_datetime(order_df["Order_Date"])
    order_df["YearMonth"] = order_df["Order_Date"].dt.to_period("M").astype(str)
    agg = order_df.groupby(["Product_ID", "YearMonth"], as_index=False)["Qty"].sum()
    # convert YearMonth -> numeric time index
    agg["YM_dt"] = pd.to_datetime(agg["YearMonth"] + "-01")
    agg["t"] = (agg["YM_dt"].dt.year - agg["YM_dt"].dt.year.min()) * 12 + (agg["YM_dt"].dt.month - agg["YM_dt"].dt.month.min())
    return agg

def forecast_product_demand(product_id: str, model_type: str = "Linear", horizon_months: int = 3):
    orders = get_orders_df()
    if orders.empty:
        return pd.DataFrame()
    feat = build_demand_features(orders[orders["Product_ID"] == product_id])
    if feat.empty or feat["t"].nunique() < 2:
        return pd.DataFrame()

    X = feat[["t"]].values
    y = feat["Qty"].values

    if model_type == "RF":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X, y)

    last_t = int(feat["t"].max())
    future_t = np.arange(last_t + 1, last_t + horizon_months + 1).reshape(-1, 1)
    preds = model.predict(future_t).clip(min=0)

    # map t back to months
    last_date = pd.to_datetime(feat["YM_dt"].max())
    future_dates = [ (last_date + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, horizon_months+1) ]
    out = pd.DataFrame({"Product_ID": product_id, "YearMonth": future_dates, "Predicted_Qty": preds.astype(int)})
    return out

def forecast_all_products(model_type: str = "Linear", horizon_months: int = 3):
    products = get_products_df()
    if products.empty:
        return pd.DataFrame()
    frames = []
    for pid in products["ID"].tolist():
        f = forecast_product_demand(pid, model_type=model_type, horizon_months=horizon_months)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    allf = pd.concat(frames, ignore_index=True)
    # add product names
    prods = products[["ID", "Name"]].rename(columns={"ID": "Product_ID"})
    allf = allf.merge(prods, on="Product_ID", how="left")
    return allf

# ==============================
# UI: LOGIN / SIGNUP
# ==============================
def auth_screen():
    st.title("🔐 Telecom Inventory Login")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        un = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if do_login(un, pw):
                st.success(f"Welcome {st.session_state.auth_user['Username']} ({st.session_state.auth_user['Role']})")
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab2:
        un2 = st.text_input("New Username")
        pw2 = st.text_input("New Password", type="password")
        role2 = st.selectbox("Role", ["Admin", "Manager", "Staff"])
        if st.button("Signup"):
            do_signup(un2, pw2, role2)

# ==============================
# UI: DASHBOARD
# ==============================
def dashboard_screen():
    st.title("📊 Dashboard")
    dashboard_kpis()

    st.subheader("📢 Notifications")
    notes = get_notifications_df()
    if notes.empty:
        st.info("No notifications.")
    else:
        st.dataframe(notes[["Message", "Created_At"]], use_container_width=True)

    st.subheader("📦 Low Stock Alerts")
    products = get_products_df()
    if not products.empty:
        low = products[products["Stock"] <= products["Reorder_Level"]]
        if low.empty:
            st.success("All good. No low-stock items.")
        else:
            st.dataframe(low[["Name", "Category", "Stock", "Reorder_Level"]], use_container_width=True)
            download_df(low, "low_stock_alerts")

    c1, c2 = st.columns([2, 1])
    with c1:
        chart_inventory_levels()
        chart_monthly_orders()
    with c2:
        chart_category_split()

    supplier_performance_table()

# ==============================
# UI: SUPPLIERS
# ==============================
def suppliers_screen(role: str):
    st.title("🏭 Suppliers")

    st.subheader("➕ Add Supplier")
    with st.form("add_supplier"):
        name = st.text_input("Name")
        contact = st.text_input("Contact")
        rating = st.number_input("Rating (1-10)", 1, 10, 7)
        perf = st.text_input("Performance Metrics (free text)")
        submitted = st.form_submit_button("Add Supplier")
    if submitted:
        add_supplier(name, contact, rating, perf)
        st.success("Supplier added.")
        st.rerun()

    st.subheader("📋 Supplier List")
    suppliers = get_suppliers_df()
    st.dataframe(suppliers, use_container_width=True)
    if not suppliers.empty:
        download_df(suppliers, "suppliers")

    if role == "Admin":
        st.divider()
        st.subheader("✏️ Update Supplier")
        if suppliers.empty:
            st.info("No suppliers.")
        else:
            sup_id = st.selectbox("Choose Supplier", suppliers["ID"])
            sup_row = suppliers[suppliers["ID"] == sup_id].iloc[0]
            with st.form("update_supplier"):
                name2 = st.text_input("Name", sup_row["Name"])
                contact2 = st.text_input("Contact", sup_row["Contact"])
                rating2 = st.number_input("Rating (1-10)", 1, 10, int(sup_row["Rating"]))
                perf2 = st.text_input("Performance Metrics", sup_row.get("Performance_Metrics", ""))
                upd = st.form_submit_button("Update")
            if upd:
                update_supplier(sup_id, name2, contact2, rating2, perf2)
                st.success("Updated.")
                st.rerun()

        st.subheader("🗑️ Delete Supplier")
        if not suppliers.empty:
            del_id = st.selectbox("Select Supplier to Delete", suppliers["ID"], key="del_sup")
            if st.button("Delete Supplier"):
                delete_supplier(del_id)
                st.success("Deleted.")
                st.rerun()

# ==============================
# UI: PRODUCTS
# ==============================
def products_screen(role: str):
    st.title("📱 Products")

    suppliers = get_suppliers_df()

    st.subheader("➕ Add Product")
    with st.form("add_product"):
        name = st.text_input("Product Name")
        category = st.text_input("Category")
        stock = st.number_input("Initial Stock", 0, step=1)
        supplier_id = st.selectbox("Supplier", suppliers["ID"]) if not suppliers.empty else None
        reorder = st.number_input("Reorder Level", 0, step=1, value=5)
        submitted = st.form_submit_button("Add Product")
    if submitted:
        if supplier_id is None:
            st.error("Please add a supplier first.")
        else:
            add_product(name, category, stock, supplier_id, reorder)
            st.success("Product added.")
            st.rerun()

    st.subheader("📋 Product List")
    products = get_products_df()
    st.dataframe(products, use_container_width=True)
    if not products.empty:
        download_df(products, "products")

    if role in ["Admin", "Manager"]:
        st.divider()
        st.subheader("✏️ Update Product")
        if products.empty:
            st.info("No products.")
        else:
            prod_id = st.selectbox("Select Product", products["ID"])
            prod_row = products[products["ID"] == prod_id].iloc[0]
            # Supplier default index
            sup_ids = suppliers["ID"].tolist() if not suppliers.empty else []
            default_idx = sup_ids.index(prod_row["Supplier_ID"]) if sup_ids and prod_row["Supplier_ID"] in sup_ids else 0
            with st.form("update_product"):
                name2 = st.text_input("Product Name", prod_row["Name"])
                category2 = st.text_input("Category", prod_row["Category"])
                stock2 = st.number_input("Stock", 0, step=1, value=int(prod_row["Stock"]))
                supplier_id2 = st.selectbox("Supplier", sup_ids, index=default_idx) if sup_ids else None
                reorder2 = st.number_input("Reorder Level", 0, step=1, value=int(prod_row["Reorder_Level"]))
                upd = st.form_submit_button("Update")
            if upd:
                update_product(prod_id, name2, category2, stock2, supplier_id2, reorder2)
                st.success("Updated.")
                st.rerun()

        if role == "Admin":
            st.subheader("🗑️ Delete Product")
            if not products.empty:
                del_id = st.selectbox("Select Product to Delete", products["ID"], key="del_prod")
                if st.button("Delete Product"):
                    delete_product(del_id)
                    st.success("Deleted.")
                    st.rerun()

# ==============================
# UI: ORDERS
# ==============================
def orders_screen(role: str):
    st.title("🧾 Orders")

    products = get_products_df()
    suppliers = get_suppliers_df()

    st.subheader("➕ Place Order (decrements stock)")
    with st.form("add_order"):
        product_id = st.selectbox("Product", products["ID"]) if not products.empty else None
        qty = st.number_input("Quantity", min_value=1, step=1, value=1)
        supplier_id = st.selectbox("Supplier", suppliers["ID"]) if not suppliers.empty else None
        unit_price = st.number_input("Unit Price (optional, for reporting)", min_value=0.0, step=0.1, value=0.0)
        submitted = st.form_submit_button("Place Order")
    if submitted:
        if product_id and supplier_id:
            try:
                place_order(product_id, qty, supplier_id, unit_price)
                st.success("Order placed and stock updated.")
            except ValueError as e:
                st.error(str(e))
            st.rerun()
        else:
            st.error("Please ensure Products and Suppliers exist.")

    st.subheader("📋 Orders List")
    orders = get_orders_df()
    if not orders.empty:
        # attach names for readability
        orders = orders.merge(products[["ID", "Name"]], left_on="Product_ID", right_on="ID", how="left", suffixes=("", "_Prod"))
        orders = orders.merge(suppliers[["ID", "Name"]].rename(columns={"Name": "Supplier_Name"}),
                              left_on="Supplier_ID", right_on="ID", how="left")
        show_cols = ["Order_Date", "Name", "Supplier_Name", "Qty", "Unit_Price"]
        show_cols = [c for c in show_cols if c in orders.columns]
        st.dataframe(orders[show_cols].sort_values("Order_Date", ascending=False), use_container_width=True)
        download_df(orders, "orders")
    else:
        st.info("No orders yet.")

# ==============================
# UI: FORECAST
# ==============================
def forecast_screen():
    st.title("🤖 Demand Forecasting")
    model_type = st.selectbox("Model", ["Linear", "RF (RandomForest)"])
    horizon = st.slider("Forecast horizon (months)", 1, 6, 3)

    if st.button("Run Forecast"):
        preds = forecast_all_products(model_type="RF" if model_type.startswith("RF") else "Linear",
                                      horizon_months=horizon)
        if preds.empty:
            st.warning("Not enough order history to forecast. Add more orders.")
            return
        st.success("Forecast completed.")
        st.dataframe(preds, use_container_width=True)
        download_df(preds, f"forecast_{model_type}_{horizon}m")

        # chart: show total forecast by month
        agg = preds.groupby("YearMonth", as_index=False)["Predicted_Qty"].sum()
        fig = px.bar(agg, x="YearMonth", y="Predicted_Qty", title="Total Forecasted Demand")
        st.plotly_chart(fig, use_container_width=True)

        # optional: select product to view
        products = get_products_df()
        product_name_map = dict(zip(products["ID"], products["Name"]))
        pids = preds["Product_ID"].unique().tolist()
        pid = st.selectbox("View forecast for product", pids, format_func=lambda x: product_name_map.get(x, x))
        if pid:
            one = preds[preds["Product_ID"] == pid]
            fig2 = px.line(one, x="YearMonth", y="Predicted_Qty", markers=True,
                           title=f"Forecast for {product_name_map.get(pid, pid)}")
            st.plotly_chart(fig2, use_container_width=True)

# ==============================
# UI: REPORTS
# ==============================
def reports_screen():
    st.title("📑 Reports & Export")

    products = get_products_df()
    suppliers = get_suppliers_df()
    orders = get_orders_df()
    st.subheader("Quick Stats")
    c1, c2, c3 = st.columns(3)
    c1.write(f"Products: **{len(products)}**")
    c2.write(f"Suppliers: **{len(suppliers)}**")
    c3.write(f"Orders: **{len(orders)}**")

    st.subheader("Export Data")
    tab1, tab2, tab3, tab4 = st.tabs(["Products", "Suppliers", "Orders", "Notifications"])
    with tab1:
        if products.empty:
            st.info("No products.")
        else:
            st.dataframe(products, use_container_width=True)
            download_df(products, "products")
    with tab2:
        if suppliers.empty:
            st.info("No suppliers.")
        else:
            st.dataframe(suppliers, use_container_width=True)
            download_df(suppliers, "suppliers")
    with tab3:
        if orders.empty:
            st.info("No orders.")
        else:
            st.dataframe(orders, use_container_width=True)
            download_df(orders, "orders")
    with tab4:
        notes = get_notifications_df()
        if notes.empty:
            st.info("No notifications.")
        else:
            st.dataframe(notes, use_container_width=True)
            download_df(notes, "notifications")

# ==============================
# APP SHELL
# ==============================
def main():
    # AUTH
    if st.session_state.auth_user is None:
        auth_screen()
        return

    user = st.session_state.auth_user
    st.sidebar.success(f"👤 {user['Username']} ({user['Role']})")
    if st.sidebar.button("Logout"):
        st.session_state.auth_user = None
        st.rerun()

    role = user["Role"]  # Admin, Manager, Staff

    # MENU
    menu = ["Dashboard", "Products", "Orders", "Forecast", "Reports"]
    if role == "Admin":
        menu.insert(1, "Suppliers")  # Admin can manage suppliers
    elif role == "Manager":
        menu.insert(1, "Suppliers")  # Manager can also manage suppliers
    # Staff: Dashboard, Products (view), Orders (place), Forecast (view), Reports (view)

    choice = st.sidebar.selectbox("Menu", menu)

    # ROLE GUARDS PER SCREEN
    if choice == "Dashboard":
        dashboard_screen()
    elif choice == "Suppliers":
        suppliers_screen(role)
    elif choice == "Products":
        products_screen(role)
    elif choice == "Orders":
        orders_screen(role)
    elif choice == "Forecast":
        forecast_screen()
    elif choice == "Reports":
        reports_screen()
    else:
        dashboard_screen()

if __name__ == "__main__":
    main()

