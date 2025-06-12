# 🚗 Spark Delivery Tracker Dashboard

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
import base64

# === THEME & STYLING ===
st.set_page_config(
    page_title="Spark Delivery Tracker",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #43a047;
        transform: scale(1.02);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 8px 12px;
    }
    .metric-box {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
    }
    .header-box {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .progress-container {
        height: 28px;
        border-radius: 14px;
        background-color: #e9ecef;
        margin: 1rem 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 14px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 14px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 1.5rem;
    }
    .highlight-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .chart-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# === CONFIG ===
TARGET_DAILY = 200
CAR_COST_MONTHLY = 620 + 120
GOOGLE_SHEET_NAME = "spark_orders"
DATA_FILE = "spark_orders.csv"
HEADERS = ["timestamp", "order_total", "miles", "earnings_per_mile", "hour"]

# === TIMEZONE ===
tz = pytz.timezone("US/Eastern")
def get_current_date():
    return datetime.now(tz).date()

# === GOOGLE SHEETS AUTH ===
try:
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    gc = gspread.authorize(creds)
    workbook = gc.open(GOOGLE_SHEET_NAME)
    worksheet = workbook.sheet1
    if not worksheet.row_values(1):
        worksheet.insert_row(HEADERS, index=1)
    try:
        user_sheet = workbook.worksheet("users")
    except gspread.WorksheetNotFound:
        user_sheet = workbook.add_worksheet(title="users", rows=100, cols=3)
        user_sheet.update("A1:C1", [["username", "password", "last_checkin_date"]])
    use_google_sheets = True
except Exception as e:
    st.error(f"Google Sheets error: {e}")
    st.warning("⚠️ Google Sheets not connected, using local fallback.")
    use_google_sheets = False

def ensure_user_row_exists(sheet, row_num):
    if not use_google_sheets:
        return
    current_rows = len(sheet.get_all_values())
    if row_num > current_rows:
        sheet.add_rows(row_num - current_rows)

def google_sheets_login():
    st.markdown("""
    <div class="header-box">
        <h1 style="color:white; margin:0; text-align:center;">🔐 Spark Delivery Tracker</h1>
        <p style="text-align:center; margin:0;">Track your deliveries and maximize earnings</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username").strip().lower()
        with col2:
            password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            users = user_sheet.get_all_records()
            for idx, user in enumerate(users):
                if user["username"].strip().lower() == username and user["password"].strip() == password:
                    today = get_current_date()
                    st.session_state.update({
                        "logged_in": True,
                        "username": username,
                        "user_row": idx + 2,
                        "last_checkin_date": user.get("last_checkin_date", "")
                    })
                    last_ci_str = st.session_state["last_checkin_date"]
                    if last_ci_str:
                        try:
                            ci_date = datetime.strptime(last_ci_str, "%Y-%m-%d").date()
                            if ci_date == today:
                                st.session_state["daily_checkin"] = {
                                    "date": today,
                                    "working": True,
                                    "goal": st.session_state.get("last_goal", TARGET_DAILY),
                                    "notes": ""
                                }
                        except:
                            pass
                    st.rerun()
            st.error("Invalid username or password")

if "logged_in" not in st.session_state:
    google_sheets_login()
    st.stop()

# === DATE FUNCTIONS ===
today = get_current_date()
yesterday = today - timedelta(days=1)

def get_date_data(df, target_date):
    # Handle empty dataframe case
    if df.empty:
        return df
        
    # Ensure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
    # Filter by date
    return df[df['timestamp'].dt.date == target_date]

# === LOAD DATA WITH SAFE DATETIME CONVERSION ===
def load_data():
    if use_google_sheets:
        try:
            records = worksheet.get_all_values()
            if not records or len(records) < 2:
                return pd.DataFrame(columns=HEADERS)
                
            headers = records[0]
            data = records[1:]
            df = pd.DataFrame(data, columns=headers)
            
            # Convert columns with error handling
            for col in ["order_total", "miles", "earnings_per_mile"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
            # Safe datetime conversion
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
            
            return df.dropna(subset=["timestamp"])
        except Exception as e:
            st.error(f"Error loading data from Google Sheets: {e}")
            return pd.DataFrame(columns=HEADERS)
    else:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            # Convert timestamp to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        return pd.DataFrame(columns=HEADERS)

# Load data and store in session state
if "df" not in st.session_state:
    st.session_state.df = load_data()
df = st.session_state.df

# === DAILY CHECK-IN FIX ===
last_checkin_str = st.session_state.get("last_checkin_date", "")
if last_checkin_str:
    try:
        last_checkin_date = datetime.strptime(last_checkin_str, "%Y-%m-%d").date()
    except ValueError:
        last_checkin_date = None
else:
    last_checkin_date = None

# If we haven't checked in today, show the check-in form
if last_checkin_date != today or "daily_checkin" not in st.session_state:
    st.markdown("""
    <div class="header-box">
        <h1 style="color:white; margin:0;">📅 Daily Check-In</h1>
    </div>
    """, unsafe_allow_html=True)
    
    working_today = st.radio("Are you working today?", ["Yes", "No"], index=0, horizontal=True)
    
    if working_today == "Yes":
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Yesterday's Performance")
            # Safely get yesterday's data
            yesterday_df = get_date_data(df, yesterday)
            earned_yest = yesterday_df["order_total"].sum() if not yesterday_df.empty else 0
            st.markdown(f"""
            <div class="metric-box">
                <div style="color: #6c757d; font-size: 14px;">Earnings</div>
                <div style="font-size: 24px; font-weight: bold; color: #343a40;">${earned_yest:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.subheader("Today's Plan")
            default_goal = st.session_state.get("last_goal", TARGET_DAILY)
            today_goal = st.number_input("💰 Today's Goal ($)", min_value=0, value=default_goal, step=10)
            notes = st.text_area("📝 Today's Notes & Goals", placeholder="Plans, goals, reminders...", height=100)
        
        if st.button("🚀 Start Tracking", type="primary", use_container_width=True):
            st.session_state["daily_checkin"] = {
                "date": today,
                "working": True,
                "goal": today_goal,
                "notes": notes
            }
            st.session_state["last_goal"] = today_goal
            st.session_state["last_checkin_date"] = str(today)
            if use_google_sheets:
                ensure_user_row_exists(user_sheet, st.session_state["user_row"])
                user_sheet.update(f"C{st.session_state['user_row']}", [[str(today)]])
            st.rerun()
    else:
        st.info("You're taking the day off. Enjoy your rest!")
        if st.button("✅ Confirm Day Off", use_container_width=True):
            st.session_state["daily_checkin"] = {
                "date": today,
                "working": False,
                "goal": 0,
                "notes": "Day off"
            }
            st.session_state["last_checkin_date"] = str(today)
            if use_google_sheets:
                ensure_user_row_exists(user_sheet, st.session_state["user_row"])
                user_sheet.update(f"C{st.session_state['user_row']}", [[str(today)]])
            st.rerun()
    st.stop()

# Initialize daily check-in if not already set
if "daily_checkin" not in st.session_state:
    st.session_state["daily_checkin"] = {
        "date": today,
        "working": True,
        "goal": st.session_state.get("last_goal", TARGET_DAILY),
        "notes": ""
    }

# === OCR ===
def extract_text_and_time(image_bytes):
    reader = easyocr.Reader(['en'])
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

# === ENHANCED OCR UPLOAD ===
def enhanced_ocr_upload():
    st.subheader("📸 Screenshot Scanner")
    with st.expander("How to use this feature"):
        st.write("1. When you see an order offer, take a screenshot")
        st.write("2. Upload it here to automatically extract details")
        st.write("3. Review and submit the extracted information")
    
    uploaded_image = st.file_uploader("Upload your screenshot", type=["jpg", "jpeg", "png"], 
                                      key="ocr_uploader", label_visibility="collapsed")
    total_auto, miles_auto, ocr_time_value = 0.0, 0.0, None
    
    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", width=300)
            
        with col2:
            with st.spinner("🔍 Analyzing image..."):
                image_bytes = BytesIO(uploaded_image.read()).getvalue()
                extracted_text, ocr_time = extract_text_and_time(image_bytes)
                total_auto, miles_auto = parse_order_details(extracted_text)
                if ocr_time:
                    try:
                        hour, minute = map(int, ocr_time.split(":"))
                        ocr_time_value = time(hour, minute)
                    except:
                        pass
            
            st.success("Data extracted successfully!")
            st.markdown(f"""
            <div class="metric-box">
                <div style="display: flex; justify-content: space-between;">
                    <div>Order Total:</div>
                    <div><strong>${total_auto if total_auto else '0.00'}</strong></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div>Miles:</div>
                    <div><strong>{miles_auto if miles_auto else '0.0'}</strong></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div>Time:</div>
                    <div><strong>{ocr_time or 'Not detected'}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("View extracted text"):
                st.code(extracted_text, language="text")
                
    return total_auto, miles_auto, ocr_time_value

# === UI ELEMENTS ===
def create_progress_bar(value, total):
    percentage = min(value / total * 100, 100) if total > 0 else 0
    return f"""
    <div style="margin-top: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>${value:.2f}</span>
            <span>${total}</span>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {percentage}%;">{percentage:.0f}%</div>
        </div>
    </div>
    """

def styled_metric(label, value, delta=None):
    delta_html = f"<div style='color: #6c757d; font-size: 14px;'>{delta}</div>" if delta else ""
    return f"""
    <div class="metric-box">
        <div style="color: #6c757d; font-size: 14px;">{label}</div>
        <div style="font-size: 24px; font-weight: bold; color: #343a40;">{value}</div>
        {delta_html}
    </div>
    """

# === MAIN APP ===
st.sidebar.markdown(f"### 👤 Welcome, {st.session_state['username']}")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Add New Delivery")

# Enhanced OCR in sidebar
total_auto, miles_auto, ocr_time_value = enhanced_ocr_upload()

# Entry form in sidebar
with st.sidebar.form("entry_form", clear_on_submit=True):
    order_total = st.number_input("Order Total ($)", min_value=0.0, step=0.01, 
                                 value=total_auto or 0.0, key="order_total_input")
    miles = st.number_input("Miles Driven", min_value=0.0, step=0.1, 
                           value=miles_auto or 0.0, key="miles_input")
    delivery_time = st.time_input("Delivery Time", 
                                 value=ocr_time_value or datetime.now(tz).time(), 
                                 key="delivery_time_input")
    submitted = st.form_submit_button("💾 Save Entry", use_container_width=True)
    
    if submitted:
        timestamp = tz.localize(datetime.combine(today, delivery_time))
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
                # Refresh data after adding
                st.session_state.df = load_data()
            else:
                # Create new dataframe with the new row
                new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                new_df.to_csv(DATA_FILE, index=False)
                st.session_state.df = new_df
            
            st.success("✅ Entry saved!")
            # Clear the uploaded image
            st.session_state.ocr_uploader = None
            # Rerun to update metrics
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error saving: {e}")

# === MAIN CONTENT ===
st.markdown("""
<div class="header-box">
    <h1 style="color:white; margin:0;">🚗 Spark Delivery Tracker</h1>
    <p style="margin:0; opacity:0.8;">Track your earnings, optimize your routes, maximize profits</p>
</div>
""", unsafe_allow_html=True)

# Display daily notes if available
if st.session_state["daily_checkin"].get("notes"):
    st.subheader("📝 Today's Notes & Goals")
    st.info(st.session_state["daily_checkin"]["notes"])

if not st.session_state["daily_checkin"]["working"]:
    st.markdown("""
    <div class="highlight-box">
        <h2 style="color:white; margin:0; text-align:center;">🏖️ Enjoy Your Day Off!</h2>
        <p style="text-align:center; margin:0;">You deserve a break - recharge for tomorrow!</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# === DATA PROCESSING ===
if not df.empty:
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Process data
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["earnings_per_mile"] = df.apply(lambda row: row["order_total"] / row["miles"] if row["miles"] and row["miles"] > 0 else 0, axis=1)

    today_df = get_date_data(df, today)
    yesterday_df = get_date_data(df, yesterday)

    today_earned = today_df["order_total"].sum() if not today_df.empty else 0
    today_miles = today_df["miles"].sum() if not today_df.empty else 0
    yesterday_earned = yesterday_df["order_total"].sum() if not yesterday_df.empty else 0

    total_earned = df["order_total"].sum()
    total_miles = df["miles"].sum()
    earnings_per_mile = total_earned / total_miles if total_miles > 0 else 0
    avg_per_day = total_earned / df["date"].nunique() if df["date"].nunique() > 0 else 0

    current_target = st.session_state["daily_checkin"].get("goal", TARGET_DAILY)

# === METRICS DISPLAY ===
if not df.empty:
    st.subheader("📊 Today's Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(styled_metric(
            "Earnings", 
            f"${today_earned:.2f}",
            f"Target: ${current_target}"
        ), unsafe_allow_html=True)
        st.markdown(create_progress_bar(today_earned, current_target), unsafe_allow_html=True)
    
    with col2:
        st.markdown(styled_metric(
            "Miles Driven", 
            f"{today_miles:.1f} mi",
            f"${today_earned/today_miles:.2f}/mi" if today_miles > 0 else ""
        ), unsafe_allow_html=True)
    
    with col3:
        car_cost_per_day = CAR_COST_MONTHLY / 30
        net_earnings = today_earned - car_cost_per_day
        st.markdown(styled_metric(
            "After Vehicle Costs", 
            f"${net_earnings:.2f}",
            f"Daily cost: ${car_cost_per_day:.2f}"
        ), unsafe_allow_html=True)

    st.subheader("📈 Historical Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(styled_metric(
        "Total Earned", 
        f"${total_earned:.2f}"
    ), unsafe_allow_html=True)
    
    col2.markdown(styled_metric(
        "Total Miles", 
        f"{total_miles:.1f} mi"
    ), unsafe_allow_html=True)
    
    col3.markdown(styled_metric(
        "Avg $ / Mile", 
        f"${earnings_per_mile:.2f}"
    ), unsafe_allow_html=True)
    
    col4.markdown(styled_metric(
        "Avg Per Day", 
        f"${avg_per_day:.2f}"
    ), unsafe_allow_html=True)

    # Yesterday comparison
    if yesterday_earned > 0:
        st.subheader("🔄 Yesterday Comparison")
        change = today_earned - yesterday_earned
        change_color = "green" if change >= 0 else "red"
        st.markdown(styled_metric(
            "Yesterday's Earnings", 
            f"${yesterday_earned:.2f}",
            f"<span style='color:{change_color};'>Today: ${change:+.2f}</span>"
        ), unsafe_allow_html=True)

    # Efficiency metrics
    if today_miles > 0 and today_earned > 0:
        mph = 30  # assumed average speed
        driving_time = today_miles / mph * 60
        hourly_rate = today_earned / (driving_time / 60) if driving_time > 0 else 0
        
        st.subheader("⏱️ Efficiency Metrics")
        col1, col2 = st.columns(2)
        col1.markdown(styled_metric(
            "Hourly Rate", 
            f"${hourly_rate:.2f}",
            f"Based on {driving_time:.0f} min driving"
        ), unsafe_allow_html=True)
        
        col2.markdown(styled_metric(
            "Fuel Efficiency", 
            f"{today_miles / (today_miles/30):.1f} MPG" if today_miles > 0 else "N/A",
            "Assuming 30 MPG"
        ), unsafe_allow_html=True)

    # CHARTS SECTION
    st.subheader("📊 Performance Charts")
    
    # Daily earnings trend
    if not df.empty:
        daily_totals = df.groupby("date")["order_total"].sum().reset_index()
        if not daily_totals.empty:
            st.markdown('<div class="chart-header">Daily Earnings Trend</div>', unsafe_allow_html=True)
            fig = px.bar(
                daily_totals, 
                x="date", 
                y="order_total", 
                text_auto='.2f',
                color="order_total",
                color_continuous_scale="tealrose"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title=None,
                yaxis_title="Earnings ($)",
                height=350
            )
            fig.add_hline(
                y=current_target, 
                line_dash="dash", 
                line_color="red", 
                annotation_text="Daily Target"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Hourly performance
        hourly_rate = df.groupby("hour")["order_total"].sum().reset_index()
        if not hourly_rate.empty:
            st.markdown('<div class="chart-header">Earnings by Hour of Day</div>', unsafe_allow_html=True)
            fig = px.bar(
                hourly_rate, 
                x="hour", 
                y="order_total", 
                color="order_total",
                color_continuous_scale="mint"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Hour of Day",
                yaxis_title="Total Earnings ($)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Smart suggestion
            best_hour = hourly_rate.loc[hourly_rate["order_total"].idxmax()]
            now = datetime.now(tz)
            hours_left = 24 - now.hour
            if hours_left > 0 and today_earned < current_target:
                needed = (current_target - today_earned) / hours_left
                st.markdown(f"""
                <div class="highlight-box">
                    <h3 style="color:white; margin:0;">🧠 Smart Suggestion</h3>
                    <p style="margin:0;">To hit your goal, earn <strong>${needed:.2f}/hr</strong> for the next {hours_left} hours</p>
                    <p style="margin:0;">Best time to work: Around <strong>{best_hour['hour']}:00</strong> (Avg ${best_hour['order_total']:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Earnings vs Miles scatter plot
        if not today_df.empty:
            st.markdown('<div class="chart-header">Today\'s Deliveries: Value vs Distance</div>', unsafe_allow_html=True)
            fig = px.scatter(
                today_df, 
                x="miles", 
                y="order_total", 
                size="order_total",
                color="earnings_per_mile",
                hover_data=["timestamp"],
                color_continuous_scale="sunset"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Miles",
                yaxis_title="Order Value ($)",
                height=400
            )
            fig.add_hline(
                y=today_df["order_total"].mean(), 
                line_dash="dash", 
                line_color="green", 
                annotation_text=f"Avg: ${today_df['order_total'].mean():.2f}"
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("📭 No data yet. Add your first delivery to get started!")

# Footer
st.markdown("---")
st.caption("Spark Delivery Tracker v2.0 | Created with Streamlit")
