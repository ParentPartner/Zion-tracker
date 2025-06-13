# 🚀 Spark Delivery Tracker with Tip Baiter Tracking (Complete Edition)

import streamlit as st
import pandas as pd
import re
from datetime import datetime, date, time, timedelta
import easyocr
import pytz
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.linear_model import LinearRegression

# === CONFIG & SETUP ===
tz = pytz.timezone("US/Eastern")
TARGET_DAILY = 200
ORDER_TYPES = ["Delivery", "Shop", "Pickup"]
PERFORMANCE_LEVELS = {
    "Excellent": {"min_epm": 3.0, "min_eph": 30},
    "Good": {"min_epm": 2.0, "min_eph": 25},
    "Fair": {"min_epm": 1.5, "min_eph": 20},
    "Poor": {"min_epm": 0, "min_eph": 0}
}

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

def add_tip_baiter_to_firestore(entry):
    db.collection("tip_baiters").add(entry)

def load_all_tip_baiters():
    data = []
    for doc in db.collection("tip_baiters").stream():
        entry = doc.to_dict()
        entry["id"] = doc.id  # Store document ID for updates/deletion
        data.append(entry)
    return pd.DataFrame(data)

# === ENHANCED OCR PARSING ===
def extract_text_from_image(image_file):
    reader = easyocr.Reader(["en"], gpu=False)
    img_bytes = image_file.read()
    image_file.seek(0)
    return reader.readtext(img_bytes, detail=0)

def parse_screenshot_text(text_list):
    joined = " ".join(text_list).lower()
    
    # Improved amount detection
    dollar_matches = re.findall(r"\$?(\d{1,3}(?:,\d{3})*\.\d{2})", joined)
    dollar_matches += re.findall(r"\$?(\d+\.\d{2})\b", joined)
    total = max([float(amt.replace(',', '')) for amt in dollar_matches]) if dollar_matches else 0.0
    
    # Miles detection
    miles = re.findall(r"(\d+(?:\.\d)?)\s?mi(?:les)?", joined)
    ml = float(miles[0]) if miles else 0.0
    
    # Time parsing
    time_match = re.search(r"\b(\d{1,2}):(\d{2})\s?([ap]m)?\b", joined, re.IGNORECASE)
    ts = datetime.now(tz)
    if time_match:
        hour, minute, period = time_match.groups()
        hour = int(hour)
        minute = int(minute)
        
        if period:
            period = period.lower()
            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
        elif hour < 6 or hour > 21:
            current_hour = ts.hour
            if current_hour < 12:  # AM
                if hour > 11:
                    hour -= 12
            else:  # PM
                if hour < 12:
                    hour += 12
        
        try:
            ts = ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            pass
    
    # Order type detection
    order_type = "Delivery"
    type_keywords = {
        "Shop": ["shop", "s&d", "shopping", "scan", "item"],
        "Pickup": ["pickup", "curbside", "pick up", "store pickup"]
    }
    
    for t, keywords in type_keywords.items():
        if any(kw in joined for kw in keywords):
            order_type = t
            break
    
    return ts, total, ml, order_type

# === AI ANALYTICS ===
def calculate_performance_metrics(df):
    metrics = {}
    metrics["total_orders"] = len(df)
    metrics["total_earnings"] = df["order_total"].sum()
    
    if "miles" in df.columns and df["miles"].sum() > 0:
        metrics["epm"] = metrics["total_earnings"] / df["miles"].sum()
    
    if "timestamp" in df.columns and len(df) > 1:
        df = df.sort_values("timestamp")
        time_diff = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
        if time_diff > 0:
            metrics["eph"] = metrics["total_earnings"] / time_diff
    
    if "order_type" in df.columns:
        metrics["type_distribution"] = df["order_type"].value_counts(normalize=True).to_dict()
    
    metrics["performance"] = "Unknown"
    if "epm" in metrics and "eph" in metrics:
        for level, criteria in PERFORMANCE_LEVELS.items():
            if metrics["epm"] >= criteria["min_epm"] and metrics["eph"] >= criteria["min_eph"]:
                metrics["performance"] = level
                break
    
    return metrics

def predict_earnings(df, target_date):
    if df.empty or "date" not in df.columns:
        return None
    
    df_daily = df.groupby("date")["order_total"].sum().reset_index()
    df_daily["date_ordinal"] = df_daily["date"].apply(lambda d: d.toordinal())
    
    if len(df_daily) < 5:
        return None
    
    X = df_daily["date_ordinal"].values.reshape(-1, 1)
    y = df_daily["order_total"].values
    model = LinearRegression().fit(X, y)
    
    target_ordinal = target_date.toordinal()
    prediction = model.predict(np.array([[target_ordinal]]))[0]
    
    return max(0, prediction)

# === TIP BAITER TRACKER ===
def tip_baiter_tracker():
    st.subheader("🚨 Tip Baiter Tracker")
    
    with st.expander("➕ Add New Tip Baiter", expanded=False):
        with st.form("tip_baiter_form"):
            name = st.text_input("Name (Required)")
            address = st.text_input("Address (Optional)")
            date_baited = st.date_input("Date", value=today)
            amount_baited = st.number_input("Amount Baited ($)", value=0.0, step=0.01)
            notes = st.text_area("Notes")
            
            if st.form_submit_button("Save Tip Baiter"):
                if not name:
                    st.error("Name is required!")
                else:
                    entry = {
                        "name": name.strip(),
                        "address": address.strip() if address else "",
                        "date": date_baited.isoformat(),
                        "amount": float(amount_baited),
                        "notes": notes.strip(),
                        "username": user,
                        "timestamp": datetime.now(tz).isoformat()
                    }
                    add_tip_baiter_to_firestore(entry)
                    st.success("Tip baiter saved!")
                    st.rerun()
    
    st.subheader("📋 Your Tip Baiters")
    tip_baiters_df = load_all_tip_baiters()
    
    if not tip_baiters_df.empty:
        tip_baiters_df = tip_baiters_df[tip_baiters_df["username"] == user]
        tip_baiters_df["date"] = pd.to_datetime(tip_baiters_df["date"])
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tip Baiters", len(tip_baiters_df))
        with col2:
            st.metric("Total Amount Baited", f"${tip_baiters_df['amount'].sum():.2f}")
        with col3:
            last_baiter = tip_baiters_df.sort_values("date", ascending=False).iloc[0]
            st.metric("Most Recent", last_baiter["date"].strftime("%m/%d/%y"))
        
        # Search and filter
        search_col, filter_col = st.columns(2)
        with search_col:
            search_term = st.text_input("Search by name or address")
        with filter_col:
            date_filter = st.selectbox("Filter by date", ["All", "Last 7 days", "Last 30 days", "Last 90 days"])
        
        # Apply filters
        if search_term:
            tip_baiters_df = tip_baiters_df[
                tip_baiters_df["name"].str.contains(search_term, case=False) |
                tip_baiters_df["address"].str.contains(search_term, case=False)
            ]
        
        if date_filter != "All":
            days = 7 if date_filter == "Last 7 days" else 30 if date_filter == "Last 30 days" else 90
            cutoff_date = today - timedelta(days=days)
            tip_baiters_df = tip_baiters_df[tip_baiters_df["date"] >= pd.to_datetime(cutoff_date)]
        
        # Display table with edit/delete options
        for _, row in tip_baiters_df.sort_values("date", ascending=False).iterrows():
            with st.expander(f"{row['name']} - {row['date'].strftime('%m/%d/%y')} - ${row['amount']:.2f}"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Address:** {row['address'] or 'Not provided'}")
                    st.write(f"**Notes:** {row['notes'] or 'No notes'}")
                with col2:
                    if st.button("🗑️ Delete", key=f"del_tb_{row['id']}"):
                        db.collection("tip_baiters").document(row["id"]).delete()
                        st.success("Tip baiter removed!")
                        st.rerun()
    else:
        st.info("No tip baiters recorded yet")

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
        
        # AI Prediction
        prediction = predict_earnings(df_all, today)
        
        col1, col2 = st.columns([2, 3])
        with col1:
            st.metric("Earnings Yesterday", f"${yesterday_sum:.2f}")
            if prediction:
                st.metric("AI Predicted Earnings", f"${prediction:.2f}")
            goal = st.number_input("Today's Goal ($)", value=TARGET_DAILY, step=10)
        with col2:
            notes = st.text_area("Notes / Mindset for today")
        if st.button("Start Tracking"):
            st.session_state["daily_checkin"] = {
                "working": True, 
                "goal": goal, 
                "notes": notes,
                "start_time": datetime.now(tz)
            }
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

# Add the Tip Baiter Tracker section
tip_baiter_tracker()

# OCR + Entry
uploaded = st.file_uploader("Upload screenshot (optional)", type=["png", "jpg", "jpeg"])
parsed = None
if uploaded:
    with st.spinner("Analyzing with AI..."):
        text_list = extract_text_from_image(uploaded)
        ts, total, ml, order_type = parse_screenshot_text(text_list)
        parsed = {
            "timestamp": ts, 
            "order_total": total,
            "miles": ml, 
            "order_type": order_type
        }
        st.success(f"AI Analysis: ${total:.2f} | {ml:.1f} mi @ {ts.strftime('%I:%M %p')} | Type: {order_type}")

with st.form("entry"):
    st.subheader("Order Entry")
    if parsed:
        default_time = parsed["timestamp"].time()
        default_date = parsed["timestamp"].date()
        default_type = parsed["order_type"]
        default_total = parsed["order_total"]
    else:
        now = datetime.now(tz)
        default_time = now.time()
        default_date = today
        default_type = "Delivery"
        default_total = 0.0
        
    selected_date = st.date_input("Date", value=default_date)
    selected_time = st.time_input("Time", value=time(default_time.hour, default_time.minute))
    order_type = st.radio("Order Type", ORDER_TYPES, index=ORDER_TYPES.index(default_type), horizontal=True)
    
    # Single total amount field
    order_total = st.number_input("Order Total ($)", value=default_total, step=0.01)
    ml = st.number_input("Miles Driven", value=parsed["miles"] if parsed else 0.0, step=0.1)

    if st.form_submit_button("Save"):
        naive_dt = datetime.combine(selected_date, selected_time)
        aware_dt = tz.localize(naive_dt)

        entry = {
            "timestamp": aware_dt.isoformat(),
            "order_total": order_total,
            "miles": ml,
            "earnings_per_mile": round(order_total/ml, 2) if ml else 0.0,
            "hour": selected_time.hour,
            "username": user,
            "order_type": order_type
        }

        add_entry_to_firestore(entry)
        st.success(f"Saved {order_type} entry at {aware_dt.strftime('%I:%M %p')}!")
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
daily_checkin = st.session_state.get("daily_checkin", {})
goal = daily_checkin.get("goal", TARGET_DAILY)
earned = today_df["order_total"].sum() if not today_df.empty else 0.0
perc = min(earned / goal * 100, 100) if goal else 0

# Calculate Earnings Per Hour
eph = None
if "start_time" in daily_checkin and not today_df.empty:
    shift_duration = (datetime.now(tz) - daily_checkin["start_time"]).total_seconds() / 3600
    if shift_duration > 0:
        eph = earned / shift_duration

# Display Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Today's Earnings", f"${earned:.2f}", f"{perc:.0f}% of ${goal} goal")
with col2:
    if eph is not None:
        st.metric("Earnings Per Hour", f"${eph:.2f}")
    else:
        st.metric("Earnings Per Hour", "Calculating...")
with col3:
    if not today_df.empty and "miles" in today_df.columns and today_df["miles"].sum() > 0:
        epm = earned / today_df["miles"].sum()
        st.metric("Earnings Per Mile", f"${epm:.2f}")

# === AI INSIGHTS ===
if not df_all.empty:
    st.subheader("🧠 AI Insights")
    
    # Performance Rating
    metrics = calculate_performance_metrics(today_df) if not today_df.empty else {}
    if "performance" in metrics:
        performance = metrics["performance"]
        color = {"Excellent": "green", "Good": "blue", "Fair": "orange", "Poor": "red"}.get(performance, "gray")
        st.markdown(f"### Performance Rating: :{color}[{performance}]")
    
    # Earnings Prediction
    prediction = predict_earnings(df_all, today + timedelta(days=1))
    if prediction:
        st.metric("Tomorrow's Prediction", f"${prediction:.2f}")
    
    # Recommendations
    st.write("### 💡 Recommendations")
    if not today_df.empty:
        if "eph" in metrics and metrics["eph"] < 20:
            st.warning("Try working during busier hours to increase your earnings per hour")
        elif "epm" in metrics and metrics["epm"] < 2.0:
            st.warning("Focus on orders with shorter distances to improve earnings per mile")
        else:
            st.success("You're doing great! Keep up the good work.")

# === DELETE ENTRIES ===
st.subheader("🗑️ Delete Entries")
selected_date = st.date_input("Select date to manage entries", value=today, key="delete_date")
entries_to_show = df_all[df_all["date"] == selected_date] if not df_all.empty else pd.DataFrame()

if not entries_to_show.empty:
    for i, row in entries_to_show.iterrows():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"🕒 {row['timestamp'].strftime('%I:%M %p')} | 💵 ${row['order_total']:.2f} | 🚗 {row['miles']} mi")
            st.caption(f"Type: {row.get('order_type', 'Delivery')}")
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

# === ANALYTICS DASHBOARD ===
st.subheader("📊 Analytics Dashboard")

if not df_all.empty:
    # Daily Earnings Trend
    st.write("### 📅 Daily Earnings Trend")
    daily_totals = df_all.groupby("date")["order_total"].sum().reset_index()
    fig = px.line(daily_totals, x="date", y="order_total", markers=True)
    fig.update_layout(xaxis_title="Date", yaxis_title="Total Earnings ($)")
    st.plotly_chart(fig, use_container_width=True)

    # Order Type Analysis
    st.write("### 📦 Order Type Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        type_counts = df_all["order_type"].value_counts().reset_index()
        fig = px.pie(type_counts, values="count", names="order_type", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        type_earnings = df_all.groupby("order_type")["order_total"].sum().reset_index()
        fig = px.bar(type_earnings, x="order_type", y="order_total", color="order_type")
        st.plotly_chart(fig, use_container_width=True)

    # Hourly Performance
    st.write("### ⏰ Hourly Earnings")
    hourly_earnings = df_all.groupby("hour")["order_total"].mean().reset_index()
    hourly_earnings["hour_str"] = hourly_earnings["hour"].apply(lambda h: f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}")
    fig = px.bar(hourly_earnings, x="hour_str", y="order_total")
    st.plotly_chart(fig, use_container_width=True)

    # Weekly Heatmap
    st.write("### 📆 Weekly Performance Heatmap")
    heat_data = df_all.groupby(["day_of_week", "hour"])["order_total"].mean().unstack()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_data = heat_data.reindex(days_order)
    fig = px.imshow(heat_data, color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Complete a few deliveries to unlock analytics")

st.caption("🧠 AI-Powered Spark Tracker v2.0 | Data stays 100% yours.")
