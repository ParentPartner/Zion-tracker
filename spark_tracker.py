# 🚗 Spark Delivery Tracker

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

# === CONFIG ===
GOOGLE_SHEET_NAME = "spark_orders"
DATA_FILE = "spark_orders.csv"
HEADERS = ["timestamp", "order_total", "miles", "earnings_per_mile", "hour"]
tz = pytz.timezone("US/Eastern")

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
    use_google_sheets = True
except Exception as e:
    st.error(f"Google Sheets error: {e}")
    st.warning("⚠️ Google Sheets not connected, using local fallback.")
    use_google_sheets = False

# === LOAD DATA ===
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
            st.error(f"Error loading data from Google Sheets: {e}")
            return pd.DataFrame(columns=HEADERS)
    else:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        return pd.DataFrame(columns=HEADERS)

if "df" not in st.session_state:
    st.session_state.df = load_data()
df = st.session_state.df

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

# === UI ===
st.title("🚗 Spark Delivery Tracker")

# === OCR UPLOAD ===
st.subheader("📸 Optional: Upload Screenshot")
uploaded_image = st.file_uploader("Upload screenshot", type=["jpg", "jpeg", "png"], key="ocr_uploader")
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
        st.text_area("OCR Output", extracted_text, height=150, key="ocr_text")

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
        timestamp = tz.localize(datetime.combine(datetime.now(tz).date(), delivery_time))
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
                st.session_state.df = load_data()
            else:
                new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                new_df.to_csv(DATA_FILE, index=False)
                st.session_state.df = new_df
            st.success("✅ Entry saved!")
            st.session_state.ocr_uploader = None
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error saving: {e}")

# === METRICS & CHARTS ===
if not df.empty:
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["earnings_per_mile"] = df.apply(lambda row: row["order_total"] / row["miles"] if row["miles"] > 0 else 0, axis=1)

    today = datetime.now(tz).date()
    today_df = df[df["date"] == today]
    yesterday_df = df[df["date"] == today - pd.Timedelta(days=1)]

    today_earned = today_df["order_total"].sum()
    today_miles = today_df["miles"].sum()
    yesterday_earned = yesterday_df["order_total"].sum() if not yesterday_df.empty else 0

    daily_totals = df.groupby("date")["order_total"].sum().reset_index()
    hourly_rate = df.groupby("hour")["order_total"].mean().reset_index()

    total_earned = df["order_total"].sum()
    total_miles = df["miles"].sum()
    earnings_per_mile = total_earned / total_miles if total_miles > 0 else 0
    avg_per_day = total_earned / df["date"].nunique() if df["date"].nunique() > 0 else 0

    st.subheader("📊 Daily Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Today's Earnings", f"${today_earned:.2f}")
    col2.metric("Today's Miles", f"{today_miles:.1f}")
    col3.metric("Yesterday's Earnings", f"${yesterday_earned:.2f}")
    col4.metric("Avg Per Day", f"${avg_per_day:.2f}")

    st.subheader("📈 Historical Performance")
    st.metric("Total Earned", f"${total_earned:.2f}")
    st.metric("Total Miles", f"{total_miles:.1f}")
    st.metric("Avg $ / Mile", f"${earnings_per_mile:.2f}")

    if not daily_totals.empty:
        fig = px.bar(daily_totals, x="date", y="order_total", text_auto=True, color="order_total", color_continuous_scale="Bluered")
        st.plotly_chart(fig, use_container_width=True)

    if not hourly_rate.empty:
        st.subheader("🕒 Average $ per Hour")
        fig = px.line(hourly_rate, x="hour", y="order_total", markers=True)
        best_hour = hourly_rate.loc[hourly_rate["order_total"].idxmax()]
        fig.add_annotation(x=best_hour["hour"], y=best_hour["order_total"],
                           text=f"Best: ${best_hour['order_total']:.2f}", showarrow=True, arrowhead=2)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data yet. Add some entries to get started!")
