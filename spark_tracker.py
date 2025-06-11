import streamlit as st
import pandas as pd
import re
from datetime import datetime
from io import BytesIO
import easyocr
import gspread
from google.oauth2.service_account import Credentials
import os
import pytz

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
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(creds)
    worksheet = gc.open(GOOGLE_SHEET_NAME).sheet1

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
        if submitted:
            if username.lower() == USERNAME.lower() and password.lower() == PASSWORD.lower():
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Invalid login")

if "logged_in" not in st.session_state:
    login()
    st.stop()

# === 🔧 Load Data (wrapped in a reusable function)
def load_data():
    if use_google_sheets:
        try:
            records = worksheet.get_all_records()
            df_local = pd.DataFrame(records)
            if 'timestamp' in df_local.columns:
                df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], errors="coerce")
            else:
                df_local = pd.DataFrame(columns=HEADERS)
        except Exception as e:
            st.error(f"Error reading from Google Sheets: {e}")
            df_local = pd.DataFrame(columns=HEADERS)
    else:
        if os.path.exists(DATA_FILE):
            df_local = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        else:
            df_local = pd.DataFrame(columns=HEADERS)
    return df_local

df = load_data()

st.title("🚗 Spark Delivery Tracker")

# === OCR Functions ===
def extract_text_and_time(image_bytes):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_bytes)

    full_text = " ".join([text for _, text, _ in results])
    top_left_time = None

    if results:
        top_left = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))[0]
        candidate_text = top_left[1].strip()
        match = re.match(r"(\d{1,2}:\d{2})", candidate_text)
        if match:
            top_left_time = match.group(1)

    return full_text, top_left_time

def parse_order_details(text):
    total, miles = None, None
    clean_text = (
        text.replace(",", "")
            .replace("O", "0")
            .replace("S", "$")
            .replace("s", "$")
    )

    lines = clean_text.lower().split("\n")
    dollar_values = re.findall(r"\$?(\d{1,4}\.\d{2})", clean_text)
    dollar_values = [float(val) for val in dollar_values if float(val) >= 5]

    for line in lines:
        if "estimate" in line:
            match = re.search(r"\$?(\d{1,4}\.\d{2})", line)
            if match:
                try:
                    total = float(match.group(1))
                    break
                except:
                    pass

    if total is None and dollar_values:
        total = max(dollar_values)

    miles_match = re.search(r"(\d+(\.\d+)?)\s*(mi|miles)", clean_text)
    if miles_match:
        miles = float(miles_match.group(1))

    return total, miles

# === OCR Upload ===
st.subheader("📸 Optional: Upload Screenshot")
uploaded_image = st.file_uploader("Upload screenshot", type=["jpg", "jpeg", "png"])
total_auto, miles_auto, ocr_time = 0.0, 0.0, None

if uploaded_image:
    with st.spinner("Reading screenshot..."):
        image_bytes = BytesIO(uploaded_image.read()).getvalue()
        extracted_text, ocr_time = extract_text_and_time(image_bytes)
        total_auto, miles_auto = parse_order_details(extracted_text)

    st.write("🧾 **Extracted Info**")
    st.write(f"**Order Total:** {total_auto if total_auto else '❌ Not found'}")
    st.write(f"**Miles:** {miles_auto if miles_auto else '❌ Not found'}")
    st.write(f"**Time (from screenshot):** {ocr_time if ocr_time else '❌ Not found'}")

    with st.expander("🧪 Show Raw OCR Text"):
        st.text_area("OCR Output", extracted_text, height=150)

# === Manual Entry Form ===
tz = pytz.timezone("US/Eastern")
now = datetime.now(tz)

with st.form("entry_form"):
    col1, col2 = st.columns(2)
    with col1:
        order_total = st.number_input("Order Total ($)", min_value=0.0, step=0.01, value=total_auto or 0.0)
    with col2:
        miles = st.number_input("Miles Driven", min_value=0.0, step=0.1, value=miles_auto or 0.0)

    try:
        if ocr_time:
            ocr_datetime = datetime.strptime(ocr_time, "%H:%M").replace(
                year=now.year, month=now.month, day=now.day
            )
        else:
            ocr_datetime = now
    except:
        ocr_datetime = now

    custom_time = st.time_input("Delivery Time", value=ocr_datetime.time())
    submitted = st.form_submit_button("Add Entry")

    if submitted:
        timestamp = now.replace(hour=custom_time.hour, minute=custom_time.minute, second=0, microsecond=0)
        earnings_per_mile = round(order_total / miles, 2) if miles > 0 else 0.0
        new_row = {
            "timestamp": timestamp.isoformat(),
            "order_total": float(order_total),
            "miles": float(miles),
            "earnings_per_mile": earnings_per_mile,
            "hour": timestamp.hour
        }

        try:
            if use_google_sheets:
                worksheet.append_row([str(new_row[k]) for k in HEADERS])
            else:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(DATA_FILE, index=False)

            st.success("✅ Entry saved!")

            # 🔧 Reload latest data before rerun
            df = load_data()
            st.experimental_rerun()

        except Exception as e:
            st.error(f"❌ Error saving entry: {e}")

# === Visualization ===
if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["earnings_per_mile"] = df.apply(lambda row: row["order_total"] / row["miles"] if row["miles"] else 0, axis=1)

    # 🔧 Sort for consistent chart order
    daily_totals = df.groupby("date")["order_total"].sum().sort_index()
    hourly_rate = df.groupby("hour")["order_total"].mean().sort_index()

    total_earned = df["order_total"].sum()
    total_miles = df["miles"].sum()
    earnings_per_mile = total_earned / total_miles if total_miles else 0
    days_logged = df["date"].nunique()
    avg_per_day = total_earned / days_logged if days_logged else 0

    last_day_total = daily_totals.iloc[-1] if not daily_totals.empty else 0
    daily_goal_remaining = max(TARGET_DAILY - last_day_total, 0)

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
        st.info("No hourly data yet. Add entries to unlock smart suggestions.")
else:
    st.info("Add some data to get started.")
