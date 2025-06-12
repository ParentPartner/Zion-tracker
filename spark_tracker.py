# 🚗 Spark Delivery Tracker (Full SQLite Version with OCR & Daily Check-In)

import streamlit as st
import pandas as pd
import sqlite3
import os
import re
from datetime import datetime, time, timedelta
from io import BytesIO
import easyocr
import pytz
import plotly.express as px

# === CONFIG ===
DB_FILE = "spark_data.db"
TARGET_DAILY = 200
CAR_COST_MONTHLY = 620 + 120
tz = pytz.timezone("US/Eastern")
HEADERS = ["timestamp", "order_total", "miles", "earnings_per_mile", "hour"]

# === DB SETUP ===
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                last_checkin_date TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS deliveries (
                timestamp TEXT,
                order_total REAL,
                miles REAL,
                earnings_per_mile REAL,
                hour INTEGER
            )
        """)

def create_default_user():
    with sqlite3.connect(DB_FILE) as conn:
        if not conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone():
            conn.execute("INSERT INTO users (username, password, last_checkin_date) VALUES (?, ?, ?)",
                         ("admin", "password", ""))

def validate_login(username, password):
    with sqlite3.connect(DB_FILE) as conn:
        return conn.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password)).fetchone()

def update_last_checkin(username, date_str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("UPDATE users SET last_checkin_date=? WHERE username=?", (date_str, username))

def get_last_checkin_date(username):
    with sqlite3.connect(DB_FILE) as conn:
        result = conn.execute("SELECT last_checkin_date FROM users WHERE username=?", (username,)).fetchone()
        return result[0] if result else ""

def add_entry_to_db(entry):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO deliveries (timestamp, order_total, miles, earnings_per_mile, hour)
            VALUES (?, ?, ?, ?, ?)
        """, (entry["timestamp"], entry["order_total"], entry["miles"], entry["earnings_per_mile"], entry["hour"]))

def load_data_from_db():
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("SELECT * FROM deliveries", conn)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df

# === HELPERS ===
def get_current_date():
    return datetime.now(tz).date()

def extract_text_from_image(image_file):
    reader = easyocr.Reader(['en'], gpu=False)
    img_bytes = image_file.read()
    image_file.seek(0)
    result = reader.readtext(img_bytes, detail=0)
    return result

def parse_screenshot_text(text):
    text_joined = " ".join(text)
    dollar_matches = re.findall(r"\$\s?(\d+(?:\.\d{1,2})?)", text_joined)
    miles_matches = re.findall(r"(\d+(?:\.\d+)?)\s?(mi|miles)", text_joined.lower())
    time_match = re.search(r"\b(\d{1,2}:\d{2})\b", text_joined)

    order_total = float(dollar_matches[0]) if dollar_matches else 0
    miles = float(miles_matches[0][0]) if miles_matches else 0

    timestamp = datetime.now(tz)
    if time_match:
        h, m = map(int, time_match.group(1).split(":"))
        timestamp = timestamp.replace(hour=h, minute=m, second=0)

    return {
        "timestamp": timestamp.isoformat(),
        "order_total": order_total,
        "miles": miles,
        "earnings_per_mile": round(order_total / miles, 2) if miles else 0,
        "hour": timestamp.hour
    }

def get_date_data(df, target_date):
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df[df['timestamp'].dt.date == target_date]

# === INIT ===
init_db()
create_default_user()

# === LOGIN ===
if "logged_in" not in st.session_state:
    st.title("🔐 Spark Tracker Login")
    with st.form("login_form"):
        username = st.text_input("Username").strip().lower()
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if validate_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.last_checkin_date = get_last_checkin_date(username)
                st.rerun()
            else:
                st.error("Invalid login. Try admin / password.")

if "logged_in" not in st.session_state:
    st.stop()

# === DAILY CHECK-IN ===
today = get_current_date()
yesterday = today - timedelta(days=1)
df = load_data_from_db()

last_ci_str = st.session_state.get("last_checkin_date", "")
last_ci_date = None
if last_ci_str:
    try:
        last_ci_date = datetime.strptime(last_ci_str, "%Y-%m-%d").date()
    except:
        pass

if last_ci_date != today:
    st.header("📅 Daily Check-In")
    working_today = st.radio("Are you working today?", ["Yes", "No"], index=0)
    if working_today == "Yes":
        yesterday_df = get_date_data(df, yesterday)
        earned_yest = yesterday_df["order_total"].sum() if not yesterday_df.empty else 0
        col1, col2 = st.columns([2, 3])
        with col1:
            st.metric("Yesterday's Earnings", f"${earned_yest:.2f}")
            default_goal = st.session_state.get("last_goal", TARGET_DAILY)
            today_goal = st.number_input("Today's Goal ($)", min_value=0, value=default_goal, step=10)
        with col2:
            notes = st.text_area("Today's Notes", placeholder="Goals, mindset, reminders", height=100)
        if st.button("Start Tracking", type="primary"):
            st.session_state["daily_checkin"] = {
                "date": today,
                "working": True,
                "goal": today_goal,
                "notes": notes
            }
            st.session_state["last_goal"] = today_goal
            st.session_state["last_checkin_date"] = str(today)
            update_last_checkin(st.session_state.username, str(today))
            st.rerun()
    else:
        if st.button("Take the day off"):
            st.session_state["daily_checkin"] = {
                "date": today,
                "working": False,
                "goal": 0,
                "notes": "Day off"
            }
            st.session_state["last_checkin_date"] = str(today)
            update_last_checkin(st.session_state.username, str(today))
            st.rerun()
    st.stop()

# === MAIN APP ===
st.title("📦 Spark Delivery Tracker")

st.markdown("Upload a screenshot or enter your order manually:")

uploaded_file = st.file_uploader("📸 Screenshot (optional)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    with st.spinner("Reading image..."):
        text = extract_text_from_image(uploaded_file)
        parsed = parse_screenshot_text(text)
        st.success(f"OCR Extracted: ${parsed['order_total']} for {parsed['miles']} mi at {parsed['timestamp'][11:16]}")
else:
    parsed = None

with st.form("manual_entry"):
    st.subheader("📝 Order Details")
    order_total = st.number_input("Order Total ($)", min_value=0.0, step=1.0,
                                  value=float(parsed['order_total']) if parsed else 0.0)
    miles = st.number_input("Miles Driven", min_value=0.0, step=0.1,
                            value=float(parsed['miles']) if parsed else 0.0)
    timestamp = st.time_input("Delivery Time", value=datetime.strptime(parsed['timestamp'][11:16], "%H:%M").time()
                              if parsed else datetime.now(tz).time())

    if st.form_submit_button("✅ Save Entry"):
        dt = datetime.combine(get_current_date(), timestamp)
        entry = {
            "timestamp": dt.isoformat(),
            "order_total": order_total,
            "miles": miles,
            "earnings_per_mile": round(order_total / miles, 2) if miles else 0,
            "hour": dt.hour
        }
        add_entry_to_db(entry)
        st.success("✅ Entry saved!")

# === METRICS & TRENDS ===
df = load_data_from_db()
today_df = get_date_data(df, today)

st.subheader("📊 Today's Progress")
earned = today_df["order_total"].sum()
goal = st.session_state.get("daily_checkin", {}).get("goal", TARGET_DAILY)
percent = min(earned / goal * 100, 100) if goal else 0
st.metric("Earnings Today", f"${earned:.2f}", f"{percent:.0f}% of goal")

col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(today_df, x="hour", y="order_total", nbins=24, title="Earnings by Hour")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(today_df, x="miles", y="order_total", title="Order Value vs Miles")
    st.plotly_chart(fig2, use_container_width=True)

st.caption("Built with ❤️ for Spark Drivers. All data stays local or in your control.")
