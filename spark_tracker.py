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
import plotly.graph_objects as go

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

# Helper functions
def get_date_data(df, target_date):
    target_df = df[df['timestamp'].dt.date == target_date]
    return target_df if not target_df.empty else pd.DataFrame(columns=df.columns)

def get_yesterday():
    tz = pytz.timezone("US/Eastern")
    return (datetime.now(tz) - timedelta(days=1)).date()

# === Daily Check-In ===
tz = pytz.timezone("US/Eastern")
today = datetime.now(tz).date()
yesterday = get_yesterday()

# Modify the daily check-in section to this:
if "last_checkin_date" not in st.session_state or st.session_state.last_checkin_date != get_current_date():
    with st.container(border=True):
        st.header("📅 Daily Check-In")
        
        working_today = st.radio(
            "Are you working today?",
            ["Yes", "No"],
            index=0
        )
        
        if working_today == "Yes":
            yesterday_df = get_date_data(df, get_yesterday())
            yesterday_earned = yesterday_df['order_total'].sum() if not yesterday_df.empty else 0
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                if yesterday_earned > 0:
                    st.metric("Yesterday's Earnings", f"${yesterday_earned:.2f}")
                else:
                    st.info("No data for yesterday")
                
                default_goal = st.session_state.get("last_goal", TARGET_DAILY)
                today_goal = st.number_input(
                    "Today's Goal ($)",
                    min_value=0,
                    value=default_goal,
                    step=10
                )
            
            with col2:
                notes = st.text_area(
                    "Today's Notes & Goals",
                    placeholder="Enter your plans, goals, or reminders for today...",
                    height=100
                )
            
            if st.button("Start Tracking", type="primary"):
                st.session_state.daily_checkin = {
                    "date": get_current_date(),
                    "working": True,
                    "goal": today_goal,
                    "notes": notes
                }
                st.session_state.last_goal = today_goal
                st.session_state.last_checkin_date = get_current_date()
                st.rerun()
        else:
            if st.button("Take the day off"):
                st.session_state.daily_checkin = {
                    "date": get_current_date(),
                    "working": False,
                    "goal": 0,
                    "notes": "Day off"
                }
                st.session_state.last_checkin_date = get_current_date()
                st.rerun()
    st.stop()
elif "daily_checkin" not in st.session_state:
    st.session_state.daily_checkin = {
        "date": get_current_date(),
        "working": False,
        "goal": 0,
        "notes": ""
    }

# Use today's custom goal if set
current_target = st.session_state.daily_checkin.get("goal", TARGET_DAILY)

# === Main App Display ===
st.title("🚗 Spark Delivery Tracker")

# Display notes prominently if they exist
if st.session_state.daily_checkin.get("notes"):
    with st.container(border=True):
        st.subheader("📝 Today's Notes")
        st.write(st.session_state.daily_checkin["notes"])

# Skip tracking if not working today
if not st.session_state.daily_checkin["working"]:
    st.success("🏖️ Enjoy your day off!")
    st.stop()

# === OCR Upload ===
st.subheader("📸 Optional: Upload Screenshot")
uploaded_image = st.file_uploader("Upload screenshot", type=["jpg", "jpeg", "png"])
total_auto, miles_auto, ocr_time = 0.0, 0.0, None
ocr_time_value = None  # Initialize time value

if uploaded_image:
    with st.spinner("Reading screenshot..."):
        image_bytes = BytesIO(uploaded_image.read()).getvalue()
        extracted_text, ocr_time = extract_text_and_time(image_bytes)
        total_auto, miles_auto = parse_order_details(extracted_text)
        
        # Convert OCR time to time object if available
        if ocr_time:
            try:
                hour, minute = map(int, ocr_time.split(':'))
                ocr_time_value = time(hour, minute)
            except ValueError:
                ocr_time_value = None

    st.write("🧾 **Extracted Info**")
    st.write(f"**Order Total:** {total_auto if total_auto else '❌ Not found'}")
    st.write(f"**Miles:** {miles_auto if miles_auto else '❌ Not found'}")
    st.write(f"**Time (from screenshot):** {ocr_time if ocr_time else '❌ Not found'}")

    with st.expander("🧪 Show Raw OCR Text"):
        st.text_area("OCR Output", extracted_text, height=150)

# === Manual Entry Form ===
with st.form("entry_form"):
    col1, col2 = st.columns(2)
    with col1:
        order_total = st.number_input("Order Total ($)", min_value=0.0, step=0.01, value=total_auto or 0.0)
    with col2:
        miles = st.number_input("Miles Driven", min_value=0.0, step=0.1, value=miles_auto or 0.0)

    # Use OCR time if available, otherwise current time
    form_default_time = ocr_time_value or datetime.now(tz).time()
    custom_time = st.time_input("Delivery Time", value=form_default_time)
    
    submitted = st.form_submit_button("Add Entry")

    if submitted:
        # Use current datetime with custom time
        timestamp = datetime.combine(datetime.now(tz).date(), custom_time)
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
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error saving entry: {e}")

# === Visualization - Enhanced with Plotly ===
if not df.empty:
    # Process data
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["earnings_per_mile"] = df.apply(lambda row: row["order_total"] / row["miles"] if row["miles"] else 0, axis=1)

    # Get date-specific data
    today_df = get_date_data(df, today)
    yesterday_df = get_date_data(df, yesterday)
    
    # Calculate metrics
    today_earned = today_df['order_total'].sum() if not today_df.empty else 0
    today_miles = today_df['miles'].sum() if not today_df.empty else 0
    yesterday_earned = yesterday_df['order_total'].sum() if not yesterday_df.empty else 0
    yesterday_miles = yesterday_df['miles'].sum() if not yesterday_df.empty else 0
    
    # More calculations
    daily_totals = df.groupby("date")["order_total"].sum().sort_index().reset_index()
    hourly_rate = df.groupby("hour")["order_total"].mean().sort_index().reset_index()
    
    total_earned = df["order_total"].sum()
    total_miles = df["miles"].sum()
    earnings_per_mile = total_earned / total_miles if total_miles else 0
    days_logged = df["date"].nunique()
    avg_per_day = total_earned / days_logged if days_logged else 0

    # === Today's Progress ===
    st.subheader("📊 Today's Progress")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Today's Earnings", f"${today_earned:.2f}", 
                 f"${today_earned - yesterday_earned:.2f} vs yesterday" if yesterday_earned > 0 else "")
    
    with col2:
        st.metric("Today's Miles", f"{today_miles:.1f}",
                 f"{today_miles - yesterday_miles:.1f} vs yesterday" if yesterday_miles > 0 else "")
    
    with col3:
        st.metric("Daily Goal", f"${current_target}", 
                 f"${current_target - today_earned:.2f} to go" if today_earned < current_target else "Goal achieved!")
    
    with col4:
        if today_earned > 0:
            progress = min(today_earned / current_target, 1)
            st.metric("Progress", f"{progress*100:.0f}%")
        else:
            st.metric("Progress", "0%")

    # Progress bar
    progress = min(today_earned / current_target, 1) if current_target > 0 else 0
    st.progress(progress, text=f"${today_earned:.2f} / ${current_target}")

    # === Yesterday Comparison ===
    if yesterday_earned > 0:
        st.subheader("🔄 Yesterday Comparison")
        yesterday_comparison = today_earned - yesterday_earned
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Yesterday's Earnings", f"${yesterday_earned:.2f}")
        
        with col2:
            st.metric("Difference", 
                      f"${yesterday_comparison:.2f}",
                      "↑ More" if yesterday_comparison > 0 else "↓ Less")

    # === Historical Data ===
    st.subheader("📈 Historical Performance")
    
    # Overall metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Earned", f"${total_earned:.2f}")
    col2.metric("Total Miles", f"{total_miles:.1f}")

    col3, col4 = st.columns(2)
    col3.metric("Avg $ / Mile", f"${earnings_per_mile:.2f}")
    col4.metric("Avg Per Day", f"${avg_per_day:.2f}")

    # === Enhanced Daily Earnings Chart ===
    if not daily_totals.empty:
        fig = px.bar(
            daily_totals,
            x="date",
            y="order_total",
            labels={"order_total": "Earnings ($)", "date": "Date"},
            text=[f"${x:.2f}" for x in daily_totals["order_total"]],
            color="order_total",
            color_continuous_scale="Bluered"
        )
        fig.update_traces(
            marker_line_color="black",
            marker_line_width=1,
            textposition="outside"
        )
        fig.update_layout(
            yaxis=dict(
                title="Earnings ($)",
                gridcolor="lightgray",
                tickprefix="$"
            ),
            xaxis=dict(
                gridcolor="lightgray",
                type="category"
            ),
            plot_bgcolor="white",
            hovermode="x"
        )
        # Add target line
        fig.add_hline(
            y=current_target,
            line_dash="dash",
            line_color="red",
            annotation_text="Daily Target",
            annotation_position="top left"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily data to show.")

    # === Enhanced Hourly Earnings Chart ===
    st.subheader("🕒 Average $ per Hour")
    if not hourly_rate.empty:
        fig = px.line(
            hourly_rate,
            x="hour",
            y="order_total",
            markers=True,
            labels={"order_total": "Avg Earnings ($)", "hour": "Hour of Day"},
            text=[f"${x:.2f}" for x in hourly_rate["order_total"]]
        )
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=10, line=dict(width=2, color="black")),
            textposition="top center"
        )
        fig.update_layout(
            yaxis=dict(
                title="Avg Earnings ($)",
                gridcolor="lightgray",
                tickprefix="$"
            ),
            xaxis=dict(
                title="Hour of Day (24h)",
                gridcolor="lightgray",
                tickvals=list(range(0, 24))
            ),
            plot_bgcolor="white",
            hovermode="x"
        )
        # Highlight best hour
        best_hour = hourly_rate.loc[hourly_rate["order_total"].idxmax()]
        fig.add_annotation(
            x=best_hour["hour"],
            y=best_hour["order_total"],
            text=f"Best hour: ${best_hour['order_total']:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hourly data to show.")

    # === Smart Suggestion ===
    st.subheader("🧠 Smart Suggestion")
    if not hourly_rate.empty:
        # Check if behind today's pace
        current_hour = datetime.now(tz).hour
        hours_remaining = 24 - current_hour
        needed_per_hour = (current_target - today_earned) / hours_remaining if hours_remaining > 0 else 0
        
        if today_earned < current_target and needed_per_hour > 0:
            st.warning(f"🚨 Behind pace! Need ${needed_per_hour:.2f}/hr to hit goal")
        
        # Original best hour suggestion
        best_hour = hourly_rate.loc[hourly_rate["order_total"].idxmax()]
        best_earning = hourly_rate.loc[hourly_rate["order_total"].idxmax()]["order_total"]
        st.success(f"**Try working around {best_hour['hour']}:00** - Highest average earnings (${best_earning:.2f}/order)")
    else:
        st.info("No hourly data yet. Add entries to unlock smart suggestions.")
else:
    st.info("Add some data to get started.")
