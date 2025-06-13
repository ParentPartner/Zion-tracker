# 🚀 Enhanced Spark Delivery Tracker (AI-Powered Edition)

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

# === AI-ENHANCED OCR PARSING ===
def extract_text_from_image(image_file):
    reader = easyocr.Reader(["en"], gpu=False)
    img_bytes = image_file.read()
    image_file.seek(0)
    return reader.readtext(img_bytes, detail=0)

def parse_screenshot_text(text_list):
    joined = " ".join(text_list).lower()
    
    # Improved dollar amount detection
    dollar_matches = re.findall(r"\$?(\d{1,3}(?:,\d{3})*\.\d{2})", joined)
    dollar_matches += re.findall(r"\$?(\d+\.\d{2})\b", joined)
    ot = 0.0
    if dollar_matches:
        try:
            # Take the largest amount found as order total
            amounts = [float(amt.replace(',', '')) for amt in dollar_matches]
            ot = max(amounts)
        except:
            ot = 0.0
    
    # Tip detection
    tip = 0.0
    tip_match = re.search(r"tip\s*\$?(\d+\.\d{2})", joined)
    if tip_match:
        tip = float(tip_match.group(1))
    
    # Improved miles detection
    miles = re.findall(r"(\d+(?:\.\d)?)\s?mi(?:les)?", joined)
    ml = float(miles[0]) if miles else 0.0
    
    # Enhanced time detection with AM/PM
    time_match = re.search(r"\b(\d{1,2}):(\d{2})\s?([ap]m)?\b", joined, re.IGNORECASE)
    ts = datetime.now(tz)
    if time_match:
        hour, minute, period = time_match.groups()
        hour = int(hour)
        minute = int(minute)
        
        # Handle 12-hour format
        if period:
            period = period.lower()
            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
        
        # Handle times without AM/PM (assume current period if between 6am-9pm)
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
            # Handle invalid time (like 25:00)
            pass
    
    # AI-enhanced order type detection
    order_type = "Delivery"  # Default
    type_keywords = {
        "Shop": ["shop", "s&d", "shopping", "scan", "item"],
        "Pickup": ["pickup", "curbside", "pick up", "store pickup"]
    }
    
    for t, keywords in type_keywords.items():
        if any(kw in joined for kw in keywords):
            order_type = t
            break
    
    # Detect if it's a batch order
    batch_order = "batch" in joined or "multiple" in joined or "2 orders" in joined
    
    return ts, ot, tip, ml, order_type, batch_order

# === AI ANALYTICS HELPERS ===
def calculate_performance_metrics(df):
    metrics = {}
    
    # Basic metrics
    metrics["total_orders"] = len(df)
    metrics["total_earnings"] = df["order_total"].sum()
    
    # Efficiency metrics
    if "miles" in df.columns and df["miles"].sum() > 0:
        metrics["epm"] = metrics["total_earnings"] / df["miles"].sum()
    
    # Time-based metrics
    if "timestamp" in df.columns and len(df) > 1:
        df = df.sort_values("timestamp")
        time_diff = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 3600
        if time_diff > 0:
            metrics["eph"] = metrics["total_earnings"] / time_diff
    
    # Order type distribution
    if "order_type" in df.columns:
        metrics["type_distribution"] = df["order_type"].value_counts(normalize=True).to_dict()
    
    # Performance rating
    metrics["performance"] = "Unknown"
    if "epm" in metrics and "eph" in metrics:
        for level, criteria in PERFORMANCE_LEVELS.items():
            if metrics["epm"] >= criteria["min_epm"] and metrics["eph"] >= criteria["min_eph"]:
                metrics["performance"] = level
                break
    
    return metrics

def predict_earnings(df, target_date):
    """Predict earnings using linear regression"""
    if df.empty or "date" not in df.columns:
        return None
    
    # Prepare data
    df_daily = df.groupby("date")["order_total"].sum().reset_index()
    df_daily["date_ordinal"] = df_daily["date"].apply(lambda d: d.toordinal())
    
    # Only predict if we have enough data
    if len(df_daily) < 5:
        return None
    
    # Train model
    X = df_daily["date_ordinal"].values.reshape(-1, 1)
    y = df_daily["order_total"].values
    model = LinearRegression().fit(X, y)
    
    # Predict for target date
    target_ordinal = target_date.toordinal()
    prediction = model.predict(np.array([[target_ordinal]]))[0]
    
    return max(0, prediction)

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
    with st.spinner("Analyzing with enhanced AI..."):
        text_list = extract_text_from_image(uploaded)
        ts, ot, tip, ml, order_type, batch_order = parse_screenshot_text(text_list)
        parsed = {
            "timestamp": ts, 
            "order_total": ot + tip,
            "base_pay": ot,
            "tip": tip,
            "miles": ml, 
            "order_type": order_type,
            "batch_order": batch_order
        }
        batch_text = " (Batch)" if batch_order else ""
        st.success(f"AI Analysis: ${ot+tip:.2f} | Base: ${ot:.2f} | Tip: ${tip:.2f} | {ml:.1f} mi @ {ts.strftime('%I:%M %p')} | Type: {order_type}{batch_text}")

with st.form("entry"):
    st.subheader("Order Entry")
    # Determine defaults - use parsed data if available, otherwise current time/today
    if parsed:
        default_time = parsed["timestamp"].time()
        default_date = parsed["timestamp"].date()
        default_type = parsed["order_type"]
        batch_default = parsed["batch_order"]
    else:
        now = datetime.now(tz)
        default_time = now.time()
        default_date = today
        default_type = "Delivery"
        batch_default = False
        
    # Add date input for manual selection
    selected_date = st.date_input("Date", value=default_date)
    
    # Create a clean default time without seconds/microseconds
    clean_default = time(default_time.hour, default_time.minute)
    selected_time = st.time_input("Time", value=clean_default)
    
    # Order type selection
    order_type = st.radio("Order Type", ORDER_TYPES, 
                          index=ORDER_TYPES.index(default_type),
                          horizontal=True)
    
    # Batch order
    batch_order = st.checkbox("Batch Order (Multiple deliveries)", value=batch_default)
    
    # Payment details
    col1, col2 = st.columns(2)
    with col1:
        base_pay = st.number_input("Base Pay ($)", value=parsed["base_pay"] if parsed else 0.0, step=0.01)
    with col2:
        tip = st.number_input("Tip ($)", value=parsed["tip"] if parsed else 0.0, step=0.01)
    
    order_total = base_pay + tip
    st.text_input("Total", value=f"${order_total:.2f}", disabled=True)
    
    ml = st.number_input("Miles Driven", value=parsed["miles"] if parsed else 0.0, step=0.1)

    if st.form_submit_button("Save"):
        # Combine selected date and time
        naive_dt = datetime.combine(selected_date, selected_time)
        aware_dt = tz.localize(naive_dt)

        entry = {
            "timestamp": aware_dt.isoformat(),
            "order_total": order_total,
            "base_pay": base_pay,
            "tip": tip,
            "miles": ml,
            "earnings_per_mile": round(order_total/ml, 2) if ml else 0.0,
            "hour": selected_time.hour,
            "username": user,
            "order_type": order_type,
            "batch_order": batch_order
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
earned = today_df["order_total"].sum() if not today_df.empty else 0.0
goal = st.session_state.get("daily_checkin", {}).get("goal", 0)
perc = min(earned / goal * 100, 100) if goal else 0

# Performance Metrics
metrics = calculate_performance_metrics(today_df) if not today_df.empty else {}

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Today's Earnings", f"${earned:.2f}", f"{perc:.0f}% of goal")
with col2:
    if "eph" in metrics:
        st.metric("Earnings Per Hour", f"${metrics['eph']:.2f}")
    else:
        st.metric("Earnings Per Hour", "-")
with col3:
    if "epm" in metrics:
        st.metric("Earnings Per Mile", f"${metrics['epm']:.2f}")
    else:
        st.metric("Earnings Per Mile", "-")

# Performance Rating
if "performance" in metrics and metrics["performance"] != "Unknown":
    performance = metrics["performance"]
    color = {"Excellent": "green", "Good": "blue", "Fair": "orange", "Poor": "red"}.get(performance, "gray")
    st.markdown(f"### Performance Rating: :{color}[{performance}]")

# === Delete Entries ===
st.subheader("🗑️ Delete Entries")
selected_date = st.date_input("Select date to manage entries", value=today)
entries_to_show = df_all[df_all["date"] == selected_date] if not df_all.empty else pd.DataFrame()

if not entries_to_show.empty:
    for i, row in entries_to_show.iterrows():
        col1, col2, col3 = st.columns([4, 2, 1])
        with col1:
            batch_text = " (Batch)" if row.get("batch_order", False) else ""
            st.write(f"🕒 {row['timestamp'].strftime('%I:%M %p')} | 💵 ${row['order_total']:.2f} | 🚗 {row['miles']} mi")
            st.caption(f"Type: {row.get('order_type', 'Delivery')}{batch_text} | Base: ${row.get('base_pay', row['order_total']):.2f} | Tip: ${row.get('tip', 0):.2f}")
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

# === AI-POWERED ANALYTICS ===
st.subheader("🧠 AI-Powered Analytics")
if not df_all.empty:
    # Performance Trends
    st.subheader("📈 Performance Trends")
    
    if "timestamp" in df_all.columns and "order_total" in df_all.columns:
        # Calculate rolling metrics
        df_daily = df_all.groupby("date")[["order_total", "miles"]].sum().reset_index()
        df_daily["epm"] = df_daily["order_total"] / df_daily["miles"]
        
        # Calculate earnings per hour
        df_all = df_all.sort_values("timestamp")
        df_time = df_all.groupby("date").agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            total_earnings=("order_total", "sum")
        ).reset_index()
        
        df_time["hours_worked"] = (df_time["end_time"] - df_time["start_time"]).dt.total_seconds() / 3600
        df_time["eph"] = df_time["total_earnings"] / df_time["hours_worked"]
        
        # Merge datasets
        df_perf = pd.merge(df_daily, df_time[["date", "eph"]], on="date", how="left")
        
        # Create plots
        fig = px.line(df_perf, x="date", y=["epm", "eph"], 
                      title="Performance Trends Over Time",
                      labels={"value": "Rate", "variable": "Metric"},
                      color_discrete_map={"epm": "blue", "eph": "green"})
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    # Order Type Efficiency Analysis
    if "order_type" in df_all.columns and "miles" in df_all.columns:
        st.subheader("📊 Order Type Efficiency")
        
        # Calculate metrics
        type_metrics = df_all.groupby("order_type").agg(
            avg_earnings=("order_total", "mean"),
            avg_miles=("miles", "mean"),
            count=("order_total", "count")
        ).reset_index()
        
        type_metrics["epm"] = type_metrics["avg_earnings"] / type_metrics["avg_miles"]
        
        # Create comparison chart
        fig = px.bar(type_metrics, x="order_type", y="epm", 
                     title="Earnings Per Mile by Order Type",
                     color="order_type",
                     labels={"epm": "Earnings Per Mile ($)"})
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Analytics
    st.subheader("🔮 Predictive Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Earnings Forecast")
        prediction = predict_earnings(df_all, today + timedelta(days=1))
        if prediction:
            st.metric("Tomorrow's Prediction", f"${prediction:.2f}")
        else:
            st.info("Need more data for prediction")
    
    with col2:
        st.write("### Best Time to Work")
        if "hour" in df_all.columns and "order_total" in df_all.columns:
            hourly_earnings = df_all.groupby("hour")["order_total"].mean().reset_index()
            best_hour = hourly_earnings.loc[hourly_earnings["order_total"].idxmax()]["hour"]
            best_time = f"{best_hour % 12 or 12} {'AM' if best_hour < 12 else 'PM'}"
            st.metric("Peak Earnings Hour", best_time)
        else:
            st.info("Need more data for analysis")
    
    # AI Recommendations
    st.subheader("🤖 Smart Recommendations")
    if "eph" in metrics and "epm" in metrics:
        eph = metrics["eph"]
        epm = metrics["epm"]
        
        recs = []
        
        # Efficiency recommendations
        if epm < 2.0:
            recs.append("⚠️ Your earnings per mile are below target. Focus on orders with shorter distances.")
        elif epm > 3.0:
            recs.append("✅ Excellent mileage efficiency! Keep prioritizing orders like these.")
        
        # Hourly rate recommendations
        if eph < 20:
            recs.append("⚠️ Your hourly earnings are low. Consider working during busier times.")
        elif eph > 30:
            recs.append("✅ Great hourly rate! You're maximizing your time effectively.")
        
        # Order type recommendations
        if "type_distribution" in metrics:
            dist = metrics["type_distribution"]
            if dist.get("Shop", 0) < 0.2:
                recs.append("ℹ️ Try accepting more Shop & Deliver orders - they often have better payouts.")
            if dist.get("Pickup", 0) > 0.4:
                recs.append("ℹ️ You're doing many Pickup orders. Check if Delivery orders might be more profitable.")
        
        if recs:
            for rec in recs:
                st.info(rec)
        else:
            st.success("Your performance is well-balanced! Keep up the good work.")
    else:
        st.info("Complete a few deliveries to get personalized recommendations")
else:
    st.info("Do a few deliveries to unlock AI insights.")

# === ADVANCED VISUALIZATIONS ===
if not df_all.empty:
    st.subheader("📊 Advanced Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Earnings Composition
        if "base_pay" in df_all.columns and "tip" in df_all.columns:
            comp_data = {
                "Base Pay": df_all["base_pay"].sum(),
                "Tips": df_all["tip"].sum()
            }
            fig = px.pie(names=list(comp_data.keys()), values=list(comp_data.values()), 
                         title="Earnings Composition", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hourly Earnings Distribution
        if "hour" in df_all.columns:
            df_all["hour_group"] = pd.cut(df_all["hour"], bins=4, 
                                          labels=["Night (12am-6am)", "Morning (6am-12pm)", 
                                                  "Afternoon (12pm-6pm)", "Evening (6pm-12am)"])
            hourly_earnings = df_all.groupby("hour_group")["order_total"].sum().reset_index()
            fig = px.bar(hourly_earnings, x="hour_group", y="order_total", 
                         title="Earnings by Time of Day",
                         labels={"order_total": "Total Earnings", "hour_group": "Time Period"})
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance Heatmap
    st.subheader("🔥 Performance Heatmap")
    if "day_of_week" in df_all.columns and "hour" in df_all.columns:
        heat_data = df_all.groupby(["day_of_week", "hour"])["order_total"].mean().unstack()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat_data = heat_data.reindex(days_order)
        
        fig = px.imshow(heat_data, 
                        labels=dict(x="Hour", y="Day", color="Avg Earnings"),
                        title="Average Earnings by Day and Hour",
                        color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

st.caption("🧠 AI-Powered Spark Tracker v2.0 | Data stays 100% yours.")
