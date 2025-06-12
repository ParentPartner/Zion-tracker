# 🚗 Spark Delivery Tracker (SQLite Version)

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

def get_current_date():
    return datetime.now(tz).date()

# === DB SETUP ===
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                last_checkin_date TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS deliveries (
                timestamp TEXT,
                order_total REAL,
                miles REAL,
                earnings_per_mile REAL,
                hour INTEGER
            )
        """)
        conn.commit()

def create_default_user():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = 'admin'")
        if not c.fetchone():
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", "password"))
            conn.commit()

def validate_login(username, password):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        return c.fetchone()

def update_last_checkin(username, date_str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("UPDATE users SET last_checkin_date=? WHERE username=?", (date_str, username))
        conn.commit()

def get_last_checkin_date(username):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT last_checkin_date FROM users WHERE username=?", (username,))
        result = c.fetchone()
        return result[0] if result else ""

def add_entry_to_db(entry):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO deliveries (timestamp, order_total, miles, earnings_per_mile, hour)
            VALUES (?, ?, ?, ?, ?)
        """, (entry["timestamp"], entry["order_total"], entry["miles"], entry["earnings_per_mile"], entry["hour"]))
        conn.commit()

def load_data_from_db():
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("SELECT * FROM deliveries", conn)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df

# === INIT ===
init_db()
create_default_user()

# === LOGIN ===
if "logged_in" not in st.session_state:
    st.title("🔐 Spark Tracker Login (SQLite)")
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

def get_date_data(df, target_date):
    if df.empty:
        return df
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df[df['timestamp'].dt.date == target_date]

last_ci_str = st.session_state.get("last_checkin_date", "")
if last_ci_str:
    try:
        last_ci_date = datetime.strptime(last_ci_str, "%Y-%m-%d").date()
    except:
        last_ci_date = None
else:
    last_ci_date = None

if last_ci_date != today or "daily_checkin" not in st.session_state:
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
            notes = st.text_area("Today's Notes & Goals", placeholder="Plans, goals, reminders...", height=100)

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

# === OCR ===
def extract_text_and_time(image_bytes):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_bytes, detail=0)
    text = "\n".join(results)
    time_match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    ocr_time = time_match.group(0) if time_match else None
    return text, ocr_time

def parse_order_details(text):
    text_lower = text.lower()
    order_total, miles = 0.0, 0.0
    estimate_match = re.search(r"\$?(\d+(?:\.\d{1,2})?)\s*estimate", text_lower)
    if estimate_match:
        order_total = float(estimate_match.group(1))
    else:
        total_matches = re.findall(r"\$?(\d+(?:\.\d{1,2})?)", text_lower)
        if total_matches:
            order_total = float(total_matches[-1])
    miles_match = re.search(r"(\d+(?:\.\d{1,2})?)\s*mi", text_lower)
    if miles_match:
        miles = float(miles_match.group(1))
    return order_total, miles

# === UI ===
st.title("🚗 Spark Delivery Tracker (SQLite)")
if st.session_state["daily_checkin"].get("notes"):
    st.subheader("📝 Today's Notes")
    st.write(st.session_state["daily_checkin"]["notes"])
if not st.session_state["daily_checkin"]["working"]:
    st.success("🏖️ Enjoy your day off!")
    st.stop()

st.subheader("📸 Optional: Upload Screenshot")
uploaded_image = st.file_uploader("Upload screenshot", type=["jpg", "jpeg", "png"], key="ocr_upload")
total_auto, miles_auto, ocr_time_value = 0.0, 0.0, None

if uploaded_image:
    with st.spinner("Reading screenshot..."):
        image_bytes = BytesIO(uploaded_image.read()).getvalue()
        extracted_text, ocr_time = extract_text_and_time(image_bytes)
        total_auto, miles_auto = parse_order_details(extracted_text)
        if ocr_time:
            try:
                hour, minute = map(int, ocr_time.split(":"))
                ocr_time_value = time(hour, minute)
            except:
                pass
    st.write("🧾 **Extracted Info**")
    st.write(f"**Order Total:** {total_auto or '❌'}")
    st.write(f"**Miles:** {miles_auto or '❌'}")
    st.write(f"**Time:** {ocr_time or '❌'}")
    with st.expander("🧪 Raw OCR Text"):
        st.text_area("OCR Output", extracted_text, height=150)

# === ENTRY FORM ===
with st.form("entry_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        order_total = st.number_input("Order Total ($)", min_value=0.0, step=0.01, value=total_auto or 0.0)
    with col2:
        miles = st.number_input("Miles Driven", min_value=0.0, step=0.1, value=miles_auto or 0.0)
    delivery_time = st.time_input("Delivery Time", value=ocr_time_value or datetime.now(tz).time())
    submitted = st.form_submit_button("Add Entry")

    if submitted:
        timestamp = tz.localize(datetime.combine(today, delivery_time))
        earnings_per_mile = round(order_total / miles, 2) if miles > 0 else 0.0
        entry = {
            "timestamp": timestamp.isoformat(),
            "order_total": float(order_total),
            "miles": float(miles),
            "earnings_per_mile": earnings_per_mile,
            "hour": timestamp.hour
        }
        add_entry_to_db(entry)
        st.success("✅ Entry saved!")
        st.rerun()

# === METRICS + CHARTS ===
df = load_data_from_db()
if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["earnings_per_mile"] = df.apply(lambda r: r["order_total"] / r["miles"] if r["miles"] > 0 else 0, axis=1)

    today_df = get_date_data(df, today)
    yesterday_df = get_date_data(df, yesterday)

    today_earned = today_df["order_total"].sum() if not today_df.empty else 0
    today_miles = today_df["miles"].sum() if not today_df.empty else 0
    yesterday_earned = yesterday_df["order_total"].sum() if not yesterday_df.empty else 0
    daily_totals = df.groupby("date")["order_total"].sum().reset_index()
    hourly_rate = df.groupby("hour")["order_total"].mean().reset_index()
    total_earned = df["order_total"].sum()
    total_miles = df["miles"].sum()
    earnings_per_mile = total_earned / total_miles if total_miles > 0 else 0
    avg_per_day = total_earned / df["date"].nunique() if df["date"].nunique() > 0 else 0
    goal = st.session_state["daily_checkin"].get("goal", TARGET_DAILY)

    st.subheader("📊 Today's Progress")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Today's Earnings", f"${today_earned:.2f}")
    col2.metric("Today's Miles", f"{today_miles:.1f}")
    col3.metric("Daily Goal", f"${goal}")
    progress = min(today_earned / goal, 1) if goal > 0 else 0
    col4.metric("Progress", f"{progress*100:.0f}%")
    st.progress(progress, text=f"${today_earned:.2f} / ${goal}")

    if yesterday_earned > 0:
        st.subheader("🔄 Yesterday Comparison")
        st.metric("Yesterday's Earnings", f"${yesterday_earned:.2f}")
        st.metric("Change", f"${today_earned - yesterday_earned:.2f}")

    st.subheader("📈 Historical Performance")
    st.metric("Total Earned", f"${total_earned:.2f}")
    st.metric("Total Miles", f"{total_miles:.1f}")
    st.metric("Avg $ / Mile", f"${earnings_per_mile:.2f}")
    st.metric("Avg Per Day", f"${avg_per_day:.2f}")

    if not daily_totals.empty:
        fig = px.bar(daily_totals, x="date", y="order_total", text_auto=True, color="order_total", color_continuous_scale="Bluered")
        fig.add_hline(y=goal, line_dash="dash", line_color="red", annotation_text="Daily Target")
        st.plotly_chart(fig, use_container_width=True)

    if not hourly_rate.empty:
        st.subheader("🕒 Average $ per Hour")
        fig = px.line(hourly_rate, x="hour", y="order_total", markers=True)
        best_hour = hourly_rate.loc[hourly_rate["order_total"].idxmax()]
        fig.add_annotation(x=best_hour["hour"], y=best_hour["order_total"],
                           text=f"Best: ${best_hour['order_total']:.2f}", showarrow=True, arrowhead=2)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 Smart Suggestion")
        now = datetime.now(tz)
        hours_left = 24 - now.hour
        if hours_left > 0 and today_earned < goal:
            needed = (goal - today_earned) / hours_left
            st.warning(f"To hit your goal, aim for ${needed:.2f}/hr for the rest of today.")
        st.success(f"🕑 Best hour: Around {best_hour['hour']}:00 — Avg ${best_hour['order_total']:.2f}")
else:
    st.info("No data yet. Add some entries to get started.")
