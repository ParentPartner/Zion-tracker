# 🚗 Spark Delivery Tracker (Firebase Edition)
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

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["firebase"])
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
    user = get_user(username)
    if not user:
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
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
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
        ts = ts.replace(hour=h, minute=m)
    return ts, ot, ml

# === STREAMLIT UI ===

# Secure login
if "logged_in" not in st.session_state:
    st.title("🔐 Spark Tracker Login")
    username = st.text_input("Username").strip().lower()
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        init_user(username)  # ensure exists (default "password")
        if validate_login(username, password):
            st.session_state.update({
                "logged_in": True,
                "username": username,
                "last_checkin_date": get_user(username).get("last_checkin_date", "")
            })
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

user = st.session_state.username
today = get_current_date()
yesterday = today - timedelta(days=1)

# Daily check-in prompt
last_ci = st.session_state.get("last_checkin_date", "")
last_ci_date = None
if last_ci:
    try:
        last_ci_date = datetime.strptime(last_ci, "%Y-%m-%d").date()
    except:
        pass

if last_ci_date != today:
    st.header("📅 Daily Check‑In")
    working = st.radio("Working today?", ("Yes", "No"))
    if working == "Yes":
        df_all = load_all_deliveries()
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
            st.experimental_rerun()
    else:
        if st.button("Take the day off"):
            st.session_state["daily_checkin"] = {"working": False, "goal": 0, "notes": "Day off"}
            st.session_state["last_checkin_date"] = today.isoformat()
            update_last_checkin(user, today.isoformat())
            st.experimental_rerun()
    st.stop()

# Main tracker interface
st.title("📦 Spark Delivery Tracker")
if st.session_state["daily_checkin"].get("working") is False:
    st.success("🏝️ Enjoy your day off!")
    st.stop()
else:
    st.markdown(st.session_state["daily_checkin"].get("notes", ""))

# OCR + Manual entry
uploaded = st.file_uploader("Upload screenshot (optional)", type=["png","jpg"])
parsed = None
if uploaded:
    with st.spinner("Analyzing…"):
        text_list = extract_text_from_image(uploaded)
        ts, ot, ml = parse_screenshot_text(text_list)
        parsed = {"timestamp": ts, "order_total": ot, "miles": ml}
        st.success(f"OCR: ${ot:.2f} | {ml:.1f} mi @ {ts.strftime('%H:%M')}")

with st.form("entry"):
    st.subheader("Order Entry")
    dt = st.time_input("Time", value=parsed["timestamp"].time() if parsed else datetime.now(tz).time())
    ot = st.number_input("Order Total ($)", value=parsed["order_total"] if parsed else 0.0, step=0.01)
    ml = st.number_input("Miles Driven", value=parsed["miles"] if parsed else 0.0, step=0.1)
    if st.form_submit_button("Save"):
        entry = {
            "timestamp": datetime.combine(today, dt).isoformat(),
            "order_total": ot,
            "miles": ml,
            "earnings_per_mile": round(ot/ml, 2) if ml else 0.0,
            "hour": dt.hour
        }
        add_entry_to_firestore(entry)
        st.success("Saved!")
        st.experimental_rerun()

# Display metrics/charts
df_all = load_all_deliveries()
df_all["date"] = df_all["timestamp"].dt.date

today_df = df_all[df_all["date"] == today]

earned = today_df["order_total"].sum()
goal = st.session_state["daily_checkin"]["goal"]
perc = min(earned / goal * 100, 100) if goal else 0
st.metric("Today's Earnings", f"${earned:.2f}", f"{perc:.0f}% of goal")

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(today_df, x="hour", y="order_total", nbins=24, title="Earnings by Hour")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.scatter(today_df, x="miles", y="order_total", title="Value vs Mileage")
    st.plotly_chart(fig, use_container_width=True)

st.caption("🏁 Built with Firebase & Streamlit | Data is secure & yours.")
