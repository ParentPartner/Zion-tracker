import streamlit as st
import pandas as pd
import easyocr
import re
from datetime import datetime
from io import BytesIO

CSV_PATH = "spark_data.csv"

def load_data():
    try:
        return pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "Time", "Total", "Miles"])

def save_data(df):
    df.to_csv(CSV_PATH, index=False)

def extract_text(image_bytes):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_bytes)
    return "\n".join([text for _, text, _ in results])  # Keep line breaks for better parsing

def parse_order_details(text):
    total, miles, order_time = None, None, None

    lines = text.lower().split("\n")

    # Try to find the $ amount just above the word "stops"
    for i, line in enumerate(lines):
        if "stops" in line and i > 0:
            prev_line = lines[i - 1]
            match = re.search(r"\$?(\d{1,4}\.\d{2})", prev_line)
            if match:
                total = float(match.group(1))
                break

    # Fallback: use highest dollar amount over $5
    if total is None:
        all_matches = re.findall(r"\$?(\d{1,4}\.\d{2})", text)
        try:
            total = max(float(m) for m in all_matches if float(m) > 5)
        except:
            pass

    # Extract miles
    miles_match = re.search(r"(\d+(\.\d+)?)\s*(mi|miles)", text.lower())
    if miles_match:
        miles = float(miles_match.group(1))

    # Extract time (usually top-left corner of screen)
    time_match = re.search(r"\b(\d{1,2}:\d{2})\b", text)
    if time_match:
        order_time = time_match.group(1)

    return total, miles, order_time

# Streamlit UI
st.set_page_config(page_title="Spark OCR Uploader", layout="centered")
st.title("📸 Spark Screenshot OCR Logger")

uploaded_image = st.file_uploader("Upload a Spark screenshot", type=["jpg", "jpeg", "png"])

if uploaded_image:
    with st.spinner("Extracting data..."):
        image_bytes = BytesIO(uploaded_image.read()).getvalue()
        extracted_text = extract_text(image_bytes)
        total, miles, phone_time = parse_order_details(extracted_text)

    st.subheader("🧾 Extracted Details")

    if total is not None:
        st.write(f"**Order Total:** ${total:.2f}")
    else:
        st.write("❌ Couldn't detect order total.")

    if miles is not None:
        st.write(f"**Miles:** {miles:.2f} mi")
    else:
        st.write("❌ Couldn't detect miles.")

    if phone_time:
        st.write(f"**Logged Time (from phone):** {phone_time}")
    else:
        phone_time = datetime.now().strftime("%H:%M")
        st.write(f"⚠️ Couldn't detect time, using current time: **{phone_time}**")

    with st.form("confirm_data"):
        total_input = st.number_input("Total", value=total or 0.0, format="%.2f")
        miles_input = st.number_input("Miles", value=miles or 0.0, format="%.2f")
        submitted = st.form_submit_button("Save to CSV")

        if submitted:
            df = load_data()
            today = datetime.now().strftime("%Y-%m-%d")
            new_row = {
                "Date": today,
                "Time": phone_time,
                "Total": total_input,
                "Miles": miles_input
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_data(df)
            st.success("✅ Entry saved!")
