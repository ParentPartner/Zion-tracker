import streamlit as st
import pandas as pd
import re
from datetime import datetime, time
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

# Helper function to get today's data
def get_today_data(df):
    tz = pytz.timezone("US/Eastern")
    today = datetime.now(tz).date()
    today_df = df[df['timestamp'].dt.date == today]
    return today_df if not today_df.empty else pd.DataFrame(columns=df.columns)

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
tz = pytz.timezone("US/Eastern")
now = datetime.now(tz)

with st.form("entry_form"):
    col1, col2 = st.columns(2)
    with col1:
        order_total = st.number_input("Order Total ($)", min_value=0.0, step=0.01, value=total_auto or 0.0)
    with col2:
        miles = st.number_input("Miles Driven", min_value=0.0, step=0.1, value=miles_auto or 0.0)

    # Use OCR time if available, otherwise current time
    form_default_time = ocr_time_value or now.time()
    custom_time = st.time_input("Delivery Time", value=form_default_time)
    
    submitted = st.form_submit_button("Add Entry")

    if submitted:
        # Use current datetime with custom time
        timestamp = datetime.combine(now.date(), custom_time)
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

    # Get today's data
    today_df = get_today_data(df)
    today_earned = today_df['order_total'].sum() if not today_df.empty else 0
    today_miles = today_df['miles'].sum() if not today_df.empty else 0
    today_goal_remaining = max(TARGET_DAILY - today_earned, 0)
    
    # Calculate metrics for all data
    daily_totals = df.groupby("date")["order_total"].sum().sort_index().reset_index()
    hourly_rate = df.groupby("hour")["order_total"].mean().sort_index().reset_index()
    
    total_earned = df["order_total"].sum()
    total_miles = df["miles"].sum()
    earnings_per_mile = total_earned / total_miles if total_miles else 0
    days_logged = df["date"].nunique()
    avg_per_day = total_earned / days_logged if days_logged else 0

    # === Today's Progress ===
    st.subheader("📊 Today's Progress")
    
    if not today_df.empty:
        progress = min(today_earned / TARGET_DAILY, 1)
        progress_text = f"${today_earned:.2f} / ${TARGET_DAILY} ({progress*100:.0f}%)"
    else:
        progress = 0
        progress_text = "No entries yet - $0.00 / $200.00 (0%)"
    
    st.progress(progress, text=progress_text)
    
    # Today's metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Today's Earnings", f"${today_earned:.2f}", "0 today" if today_earned == 0 else "")
    col2.metric("Today's Miles", f"{today_miles:.1f}", "0 today" if today_miles == 0 else "")
    col3.metric("Daily Goal", f"${TARGET_DAILY}", f"${today_goal_remaining:.2f} to go")

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
            y=TARGET_DAILY,
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
        hours_remaining = 24 - datetime.now(pytz.timezone("US/Eastern")).hour
        needed_per_hour = today_goal_remaining / hours_remaining if hours_remaining > 0 else 0
        
        if today_earned < TARGET_DAILY and needed_per_hour > 0:
            st.warning(f"🚨 Behind pace! Need ${needed_per_hour:.2f}/hr to hit goal")
        
        # Original best hour suggestion
        best_hour = hourly_rate.loc[hourly_rate["order_total"].idxmax()]["hour"]
        best_earning = hourly_rate.loc[hourly_rate["order_total"].idxmax()]["order_total"]
        st.success(f"**Try working around {best_hour}:00** - Highest average earnings (${best_earning:.2f}/order)")
    else:
        st.info("No hourly data yet. Add entries to unlock smart suggestions.")
else:
    st.info("Add some data to get started.")
