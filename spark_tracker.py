import streamlit as st
import pandas as pd
import re
from datetime import datetime
from io import BytesIO
import easyocr
import gspread
from google.oauth2.service_account import Credentials
import os

# === Config ===
TARGET_DAILY = 200
CAR_COST_MONTHLY = 620 + 120
GOOGLE_SHEET_NAME = "spark_orders"
DATA_FILE = "spark_orders.csv"
HEADERS = ["timestamp", "order_total", "miles", "earnings_per_mile", "hour"]

# === Auth ===
USERNAME = st.secrets["SPARK_USER"]
PASSWORD = st.secrets["SPARK_PASS"]

# === Google Sheets Auth ===
try:
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    gc = gspread.authorize(creds)
    worksheet = gc.open(GOOGLE_SHEET_NAME).sheet1

    # ✅ Ensure headers are present
    if not worksheet.row_values(1):
        worksheet.insert_row(HEADERS, index=1)

    use_google_sheets = True
except Exception as e:
    st.error(f"Google Sheets error: {e}")
    st.warning("⚠️ Google Sheets not connected, using local CSV fallback.")
    use_google_sheets = False

# === Login ===
def login():
    st.title("🔐 Spark Tracker Login")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted and username.lower() == USERNAME.lower() and password.lower() == PASSWORD.lower():
            st.session_state["logged_in"] = True
            st.rerun()
        elif submitted:
            st.error("Invalid login")

if "logged_in" not in st.session_state:
    login()
    st.stop()

# === Load Data ===
if use_google_sheets:
    records = worksheet.get_all_records()
    df = pd.DataFrame(records)
    if not df.empty and 'timestamp' in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif df.empty:
        df = pd.DataFrame(columns=HEADERS)
else:
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    else:
        df = pd.DataFrame(columns=HEADERS)

# === App Title ===
st.title("🚗 Spark Delivery Tracker")

# === OCR Logic ===
def extract_text(image_bytes):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_bytes)
    return " ".join([text for _, text, _ in results])

def parse_order_details(text):
    total, miles, order_time = None, None, None
    clean_text = text.replace(",", "").replace("O", "0")
    lines = clean_text.lower().split("\n")
    total_keywords = ["earnings", "order total", "total", "payout", "you earned", "tip"]

    for i, line in enumerate(lines):
        if any(keyword in line for keyword in total_keywords):
            match = re.search(r"\$?(\d{1,4}\.\d{2})", line)
            if match:
                total = float(match.group(1))
                break
        elif i + 1 < len(lines) and any(keyword in lines[i + 1] for keyword in total_keywords):
            match = re.search(r"\$?(\d{1,4}\.\d{2})", line)
            if match:
                total = float(match.group(1))
                break

    if total is None:
        all_matches = re.findall(r"\$?(\d{1,4}\.\d{2})", clean_text)
        if all_matches:
            try:
                total = max(float(val) for val in all_matches if float(val) >= 5)
            except:
                pass

    miles_match = re.search(r"(\d+(\.\d+)?)\s*(mi|miles)", clean_text)
    if miles_match:
        miles = float(miles_match.group(1))

    time_match = re.search(r"\b(\d{1,2}:\d{2})\b", clean_text)
    if time_match:
        order_time = time_match.group(1)

    return total, miles, order_time

# === Optional OCR Upload ===
st.subheader("📸 Optional: Upload Screenshot Instead")

uploaded_image = st.file_uploader("Drag & drop or browse for screenshot", type=["jpg", "jpeg", "png"])
total_auto, miles_auto = 0.0, 0.0

if uploaded_image:
    with st.spinner("Reading screenshot..."):
        image_bytes = BytesIO(uploaded_image.read()).getvalue()
        extracted_text = extract_text(image_bytes)
        total_auto, miles_auto, _ = parse_order_details(extracted_text)

    st.write("🧾 **Extracted Info**")
    st.write(f"**Order Total:** {total_auto if total_auto else '❌ Not found'}")
    st.write(f"**Miles:** {miles_auto if miles_auto else '❌ Not found'}")

# === Manual Entry Form (Auto-filled if image uploaded) ===
with st.form("entry_form"):
    col1, col2 = st.columns(2)
    with col1:
        order_total = st.number_input("Order Total ($)", min_value=0.0, step=0.01, value=total_auto)
    with col2:
        miles = st.number_input("Miles Driven", min_value=0.0, step=0.1, value=miles_auto)
    submitted = st.form_submit_button("Add Entry")
    if submitted:
        now = datetime.now()
        new_row = {
            "timestamp": now.isoformat(),
            "order_total": order_total,
            "miles": miles,
            "earnings_per_mile": round(order_total / miles, 2) if miles > 0 else 0,
            "hour": now.hour
        }
        if use_google_sheets:
            worksheet.append_row(list(new_row.values()))
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
        st.success("✅ Entry saved!")

# === Metrics + Charts ===
if not df.empty and "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["earnings_per_mile"] = df["order_total"] / df["miles"].replace(0, pd.NA)

    daily_totals = df.groupby("date")["order_total"].sum()
    hourly_rate = df.groupby("hour")["order_total"].mean()

    total_earned = df["order_total"].sum()
    total_miles = df["miles"].sum()
    earnings_per_mile = total_earned / total_miles if total_miles else 0
    days_logged = df["date"].nunique()
    avg_per_day = total_earned / days_logged if days_logged else 0
    daily_goal_remaining = max(TARGET_DAILY - daily_totals.iloc[-1], 0) if not daily_totals.empty else TARGET_DAILY

    col1, col2 = st.columns(2)
    col1.metric("Total Earned", f"${total_earned:.2f}")
    col2.metric("Total Miles", f"{total_miles:.1f}")

    col3, col4 = st.columns(2)
    col3.metric("Avg $ / Mile", f"${earnings_per_mile:.2f}")
    col4.metric("Avg Per Day", f"${avg_per_day:.2f}")

    st.markdown(f"🧾 **Today's Goal Left:** ${daily_goal_remaining:.2f}")

    st.subheader("📈 Daily Earnings")
    st.bar_chart(daily_totals)

    st.subheader("🕒 Hourly $ / Order")
    st.line_chart(hourly_rate)

    st.subheader("🧠 Smart Suggestion")
    if not hourly_rate.empty:
        best_hour = hourly_rate.idxmax()
        st.success(f"Try working more around **{best_hour}:00** — that's your highest earning hour!")
    else:
        st.info("No hourly data available yet. Add some entries to get smart suggestions.")


else:
    st.info("Add some data to get started.")
