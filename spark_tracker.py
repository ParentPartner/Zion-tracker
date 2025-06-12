# 🚗 Spark Delivery Tracker (Firebase Edition with 12-Hour Format & Enhanced Visuals)

import streamlit as st
import pandas as pd
import re
from datetime import datetime, date, time, timedelta
from io import BytesIO
import easyocr
import pytz
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, firestore

# === CONFIG & SETUP ===
tz = pytz.timezone("US/Eastern")
TARGET_DAILY = 200

if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)
db = firestore.client()

def get_current_date() -> date:
    return datetime.now(tz).date()

# === FIRESTORE HELPERS ===
def get_user(username):
    doc = db.collection("users").document(username).get()
    return doc.to_dict() if doc.exists else None

def validate_login(username, password):
    user = get_user(username)
    return user and user.get("password") == password

def update_last_checkin(username, date_str):
    db.collection("users").document(username).update({"last_checkin_date": date_str})

def init_user(username, password="password"):
    if not get_user(username):
        db.collection("users").document(username).set({
            "password": password,
            "last_checkin_date": ""
        })

def add_entry_to_firestore(entry):
    db.collection("deliveries").add(entry)

def load_all_deliveries():
    data = []
    for doc in db.collection("deliveries").stream():
        data.append(doc.to_dict())
    df = pd.DataFrame(data)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

# === OCR PARSING ===
def extract_text_from_image(image_file):
    reader = easyocr.Reader(["en"], gpu=False)
    img_bytes = image_file.read()
    image_file.seek(0)
    return reader.readtext(img_bytes, detail=0)

def parse_screenshot_text(text_list):
    joined = " ".join(text_list).lower()
    dollar = re.search(r"\$?(\d+(?:\.\d{1,2}))", joined)
    miles = re.search(r"(\d+(?:\.\d))\s?mi", joined)
    time_match = re.search(r"\b(\d{1,2}:\d{2})\b", joined)

    ot = float(dollar.group(1)) if dollar else 0.0
    ml = float(miles.group(1)) if miles else 0.0
    ts = datetime.now(tz)
    if time_match:
        h, m = map(int, time_match.group(1).split(":"))
        ts = ts.replace(hour=h, minute=m, second=0, microsecond=0)
    return ts, ot, ml

# === STREAMLIT UI ===
if "logged_in" not in st.session_state:
    st.title("🔐 Spark Tracker Login")
    username = st.text_input("Username").strip().lower()
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        init_user(username)
        if validate_login(username, password):
            st.session_state.update({
                "logged_in": True,
                "username": username,
                "last_checkin_date": get_user(username).get("last_checkin_date", "")
            })
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

user = st.session_state.username
today = get_current_date()
yesterday = today - timedelta(days=1)

# Daily Check-in
last_ci = st.session_state.get("last_checkin_date", "")
last_ci_date = datetime.strptime(last_ci, "%Y-%m-%d").date() if last_ci else None

if last_ci_date != today:
    st.header("📅 Daily Check‑In")
    working = st.radio("Working today?", ("Yes", "No"))
    if working == "Yes":
        df_all = load_all_deliveries()
        yesterday_sum = 0
        if not df_all.empty and "timestamp" in df_all.columns:
            df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
            yesterday_sum = df_all[df_all["timestamp"].dt.date == yesterday]["order_total"].sum()
        col1, col2 = st.columns([2, 3])
        with col1:
            st.metric("Earnings Yesterday", f"${yesterday_sum:.2f}")
            goal = st.number_input("Today's Goal ($)", value=TARGET_DAILY, step=10)
        with col2:
            notes = st.text_area("Notes / Mindset for today")
        if st.button("Start Tracking"):
            st.session_state["daily_checkin"] = {"working": True, "goal": goal, "notes": notes}
            st.session_state["last_checkin_date"] = today.isoformat()
            update_last_checkin(user, today.isoformat())
            st.rerun()
    else:
        if st.button("Take the day off"):
            st.session_state["daily_checkin"] = {"working": False, "goal": 0, "notes": "Day off"}
            st.session_state["last_checkin_date"] = today.isoformat()
            update_last_checkin(user, today.isoformat())
            st.rerun()
    st.stop()

# Main Interface
st.title("📦 Spark Delivery Tracker")
if st.session_state.get("daily_checkin", {}).get("working") is False:
    st.success("🏝️ Enjoy your day off!")
    st.stop()
else:
    st.markdown(st.session_state.get("daily_checkin", {}).get("notes", ""))

# OCR + Entry
uploaded = st.file_uploader("Upload screenshot (optional)", type=["png", "jpg", "jpeg"])
parsed = None
if uploaded:
    with st.spinner("Analyzing…"):
        text_list = extract_text_from_image(uploaded)
        ts, ot, ml = parse_screenshot_text(text_list)
        parsed = {"timestamp": ts, "order_total": ot, "miles": ml}
        st.success(f"OCR: ${ot:.2f} | {ml:.1f} mi @ {ts.strftime('%I:%M %p')}")

with st.form("entry"):
    st.subheader("Order Entry")
    dt = st.time_input("Time", value=parsed["timestamp"].time() if parsed else datetime.now(tz).time())
    ot = st.number_input("Order Total ($)", value=parsed["order_total"] if parsed else 0.0, step=0.01)
    ml = st.number_input("Miles Driven", value=parsed["miles"] if parsed else 0.0, step=0.1)
    if st.form_submit_button("Save"):
        entry = {
            "timestamp": tz.localize(datetime.combine(today, dt)).isoformat(),
            "order_total": ot,
            "miles": ml,
            "earnings_per_mile": round(ot/ml, 2) if ml else 0.0,
            "hour": dt.hour,
            "username": user
        }
        add_entry_to_firestore(entry)
        st.success("Saved!")
        st.rerun()

# Load + Filter
df_all = load_all_deliveries()
if not df_all.empty and "timestamp" in df_all.columns:
    df_all = df_all[df_all["username"] == user] if "username" in df_all.columns else pd.DataFrame()
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    df_all = df_all.dropna(subset=["timestamp"])
    df_all["date"] = df_all["timestamp"].dt.date
    df_all["hour"] = df_all["timestamp"].dt.hour
    df_all["hour_12"] = df_all["timestamp"].dt.strftime("%I %p")
    df_all["day_of_week"] = df_all["timestamp"].dt.day_name()
    today_df = df_all[df_all["date"] == today]
else:
    df_all = pd.DataFrame()
    today_df = pd.DataFrame()

# Earnings Goal
earned = today_df["order_total"].sum() if not today_df.empty else 0.0
goal = st.session_state.get("daily_checkin", {}).get("goal", 0)
perc = min(earned / goal * 100, 100) if goal else 0
st.metric("Today's Earnings", f"${earned:.2f}", f"{perc:.0f}% of goal")

# === Delete Entries ===
st.subheader("🗑️ Delete Entries")

selected_date = st.date_input("Select date to manage entries", value=today)
entries_to_show = df_all[df_all["date"] == selected_date] if not df_all.empty else pd.DataFrame()

if not entries_to_show.empty:
    for i, row in entries_to_show.iterrows():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"🕒 {row['timestamp'].strftime('%I:%M %p')} | 💵 ${row['order_total']:.2f} | 🚗 {row['miles']} mi")
        with col2:
            st.write(f"EPM: ${row['earnings_per_mile']:.2f}")
        with col3:
            if st.button("🗑️ Delete", key=f"del_{i}"):
                all_docs = list(db.collection("deliveries").stream())
                for doc in all_docs:
                    data = doc.to_dict()
                    if (
                        data.get("username") == user and
                        abs(pd.to_datetime(data.get("timestamp")) - row["timestamp"]) < timedelta(seconds=5) and
                        float(data.get("order_total")) == row["order_total"]
                    ):
                        db.collection("deliveries").document(doc.id).delete()
                        st.success("Entry deleted!")
                        st.rerun()
else:
    st.info("No entries found for this date.")

# === Analytics ===
st.subheader("📈 Analytics & Trends")
col1, col2 = st.columns(2)

if not df_all.empty:
    with col1:
        daily_totals = df_all.groupby("date")["order_total"].sum().reset_index()
        fig = px.line(daily_totals, x="date", y="order_total", title="📅 Daily Earnings", markers=True)
        fig.update_layout(xaxis_title="Date", yaxis_title="Total Earned ($)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        avg_by_hour = df_all.groupby("hour_12")["order_total"].mean().reset_index()
        fig = px.bar(avg_by_hour, x="hour_12", y="order_total", title="⏰ Avg Earnings by Hour",
                     labels={"hour_12": "Hour", "order_total": "Avg $"})
        fig.update_layout(xaxis_categoryorder="array", xaxis_categoryarray=avg_by_hour["hour_12"],
                          template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.header("🤖 Smart Suggestions")
    hour_avg = df_all.groupby("hour")["order_total"].mean()
    best_hour = hour_avg.idxmax()
    best_hour_val = hour_avg.max()

    weekday_avg = df_all.groupby("day_of_week")["order_total"].mean()
    best_day = weekday_avg.idxmax()
    best_day_val = weekday_avg.max()

    df_all["efficiency"] = df_all["order_total"] / df_all["miles"].replace(0, 0.01)
    best_eff = df_all["efficiency"].median()

    st.markdown(f"""
    ✅ Based on your delivery history:

    - **📆 Best day:** `{best_day}` – avg **${best_day_val:.2f}**/order  
    - **⏰ Best hour:** `{best_hour % 12 or 12}:00 {'AM' if best_hour < 12 else 'PM'}` – avg **${best_hour_val:.2f}**  
    - **🚗 Efficiency Tip:** Aim for **${best_eff:.2f}/mile**
    """)

    if best_day_val > 25 and best_hour_val > 20:
        st.success(f"🔥 Tip: Deliver on **{best_day} between {best_hour % 12 or 12}:00 {'AM' if best_hour < 12 else 'PM'} and {(best_hour + 1) % 12 or 12}:00**")

    st.subheader("📅 Hourly Earnings by Weekday")
    dow_summary = df_all.groupby(["day_of_week", "hour"])["order_total"].mean().reset_index()
    dow_summary["hour"] = dow_summary["hour"].astype(int)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_summary["day_of_week"] = pd.Categorical(dow_summary["day_of_week"], categories=dow_order, ordered=True)
    pivot = dow_summary.pivot(index="hour", columns="day_of_week", values="order_total").fillna(0)
    fig = px.imshow(pivot, labels=dict(x="Day", y="Hour (24h)", color="Avg $"), title="📊 Heatmap of Avg Earnings")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Do a few deliveries to unlock smart insights.")

st.caption("🏁 Built with Firebase & Streamlit | Data stays 100% yours.")
