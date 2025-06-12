import streamlit as st
import pandas as pd
import re
from datetime import datetime, time, timedelta
from io import BytesIO
import easyocr
import gspread
from google.oauth2.service_account import Credentials
import os
import pytz
import plotly.express as px

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
                st.experimental_rerun()
            else:
                st.error("Invalid login")

if "logged_in" not in st.session_state:
    login()
    st.stop()

# === Load Data ===
def load_data():
    if use_google_sheets:
        try:
            records = worksheet.get_all_values()
            if not records or len(records) < 2:
                return pd.DataFrame(columns=HEADERS)
            headers = records[0]
            data = records[1:]
            df = pd.DataFrame(data, columns=headers)
            for col in ["order_total", "miles", "earnings_per_mile"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
            return df.dropna(subset=["timestamp"])
        except Exception as e:
            st.error(f"Error loading from Google Sheets: {e}")
            return pd.DataFrame(columns=HEADERS)
    else:
        if os.path.exists(DATA_FILE):
            return pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
        return pd.DataFrame(columns=HEADERS)

df = load_data()

# === Helper Functions ===
tz = pytz.timezone("US/Eastern")
def get_current_date():
    return datetime.now(tz).date()

def get_yesterday():
    return (datetime.now(tz) - timedelta(days=1)).date()

def get_date_data(df, target_date):
    return df[df['timestamp'].dt.date == target_date]

def extract_text_and_time(image_bytes):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_bytes, detail=0)
    text = "\n".join(results)

    time_match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    ocr_time = time_match.group(0) if time_match else None
    return text, ocr_time

def parse_order_details(text):
    text_lower = text.lower()
    order_total = 0.0
    miles = 0.0

    # Look for $XX.XX estimate pattern first
    estimate_pattern = re.compile(r"\$?(\d+(?:\.\d{1,2})?)\s*estimate")
    estimate_match = estimate_pattern.search(text_lower)
    if estimate_match:
        order_total = float(estimate_match.group(1))
    else:
        # fallback: any dollar amount in text
        total_matches = re.findall(r"\$?(\d+(?:\.\d{1,2})?)", text_lower)
        if total_matches:
            order_total = float(total_matches[-1])

    miles_match = re.search(r"(\d+(?:\.\d{1,2})?)\s*mi", text_lower)
    if miles_match:
        miles = float(miles_match.group(1))

    return order_total, miles

# === Daily Check-In ===
today = get_current_date()
yesterday = get_yesterday()

if "last_checkin_date" not in st.session_state or st.session_state["last_checkin_date"] != today:
    with st.container():
        st.header("📅 Daily Check-In")
        working_today = st.radio("Are you working today?", ["Yes", "No"], index=0)

        if working_today == "Yes":
            yesterday_df = get_date_data(df, yesterday)
            yesterday_earned = yesterday_df["order_total"].sum() if not yesterday_df.empty else 0
            col1, col2 = st.columns([2, 3])

            with col1:
                if yesterday_earned > 0:
                    st.metric("Yesterday's Earnings", f"${yesterday_earned:.2f}")
                else:
                    st.info("No data for yesterday")

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
                st.session_state["last_checkin_date"] = today
                st.experimental_rerun()

        else:
            if st.button("Take the day off"):
                st.session_state["daily_checkin"] = {
                    "date": today,
                    "working": False,
                    "goal": 0,
                    "notes": "Day off"
                }
                st.session_state["last_checkin_date"] = today
                st.experimental_rerun()
    st.stop()
elif "daily_checkin" not in st.session_state:
    # Initialize with a default day off if somehow missing
    st.session_state["daily_checkin"] = {
        "date": today,
        "working": False,
        "goal": 0,
        "notes": ""
    }

current_target = st.session_state["daily_checkin"].get("goal", TARGET_DAILY)
st.title("🚗 Spark Delivery Tracker")

# === Notes Display ===
if st.session_state["daily_checkin"].get("notes"):
    with st.container():
        st.subheader("📝 Today's Notes")
        st.write(st.session_state["daily_checkin"]["notes"])

if not st.session_state["daily_checkin"]["working"]:
    st.success("🏖️ Enjoy your day off!")
    st.stop()

# === OCR Upload ===
st.subheader("📸 Optional: Upload Screenshot")
uploaded_image = st.file_uploader("Upload screenshot", type=["jpg", "jpeg", "png"])
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
            except ValueError:
                pass

    st.write("🧾 **Extracted Info**")
    st.write(f"**Order Total:** {total_auto if total_auto > 0 else '❌ Not found'}")
    st.write(f"**Miles:** {miles_auto if miles_auto > 0 else '❌ Not found'}")
    st.write(f"**Time (from screenshot):** {ocr_time or '❌ Not found'}")
    with st.expander("🧪 Show Raw OCR Text"):
        st.text_area("OCR Output", extracted_text, height=150)

# === Manual Entry ===
with st.form("entry_form"):
    col1, col2 = st.columns(2)
    with col1:
        order_total = st.number_input("Order Total ($)", min_value=0.0, step=0.01, value=total_auto or 0.0)
    with col2:
        miles = st.number_input("Miles Driven", min_value=0.0, step=0.1, value=miles_auto or 0.0)

    delivery_time = st.time_input("Delivery Time", value=ocr_time_value or datetime.now(tz).time())
    submitted = st.form_submit_button("Add Entry")

    if submitted:
        timestamp = datetime.combine(today, delivery_time)
        timestamp = tz.localize(timestamp)
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
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ Error saving entry: {e}")

# === Visualization ===
if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["earnings_per_mile"] = df.apply(lambda row: row["order_total"] / row["miles"] if row["miles"] else 0, axis=1)

    today_df = get_date_data(df, today)
    yesterday_df = get_date_data(df, yesterday)

    today_earned = today_df["order_total"].sum() if not today_df.empty else 0
    today_miles = today_df["miles"].sum() if not today_df.empty else 0
    yesterday_earned = yesterday_df["order_total"].sum() if not yesterday_df.empty else 0
    yesterday_miles = yesterday_df["miles"].sum() if not yesterday_df.empty else 0

    daily_totals = df.groupby("date")["order_total"].sum().reset_index()
    hourly_rate = df.groupby("hour")["order_total"].mean().reset_index()

    total_earned = df["order_total"].sum()
    total_miles = df["miles"].sum()
    earnings_per_mile = total_earned / total_miles if total_miles else 0
    avg_per_day = total_earned / df["date"].nunique()

    # === Progress ===
    st.subheader("📊 Today's Progress")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Today's Earnings", f"${today_earned:.2f}")
    col2.metric("Today's Miles", f"{today_miles:.1f}")
    col3.metric("Daily Goal", f"${current_target}")
    progress = min(today_earned / current_target, 1) if current_target > 0 else 0
    col4.metric("Progress", f"{progress*100:.0f}%")
    st.progress(progress, text=f"${today_earned:.2f} / ${current_target}")

    if yesterday_earned > 0:
        st.subheader("🔄 Yesterday Comparison")
        st.metric("Yesterday's Earnings", f"${yesterday_earned:.2f}")
        st.metric("Change", f"${today_earned - yesterday_earned:.2f}")

    st.subheader("📈 Historical Performance")
    col1, col2 = st.columns(2)
    col1.metric("Total Earned", f"${total_earned:.2f}")
    col2.metric("Total Miles", f"{total_miles:.1f}")
    col3, col4 = st.columns(2)
    col3.metric("Avg $ / Mile", f"${earnings_per_mile:.2f}")
    col4.metric("Avg Per Day", f"${avg_per_day:.2f}")

    # === Charts ===
    if not daily_totals.empty:
        fig = px.bar(daily_totals, x="date", y="order_total", text_auto=True, color="order_total", color_continuous_scale="Bluered")
        fig.update_layout(title="Daily Earnings", yaxis_title="Earnings ($)")
        fig.add_hline(y=current_target, line_dash="dash", line_color="red", annotation_text="Daily Target")
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
        hours_remaining = 24 - now.hour
        if hours_remaining > 0 and today_earned < current_target:
            needed = (current_target - today_earned) / hours_remaining
            st.warning(f"To hit your goal, try earning ${needed:.2f}/hr for the rest of today.")
        st.success(f"🕑 Best time to work: Around {best_hour['hour']}:00 - Avg ${best_hour['order_total']:.2f}/order")
else:
    st.info("No data yet. Add some entries to get started!")
