# 🚀 Spark Delivery Tracker with Incentives & Tip Baiter Tracking (Complete Edition)

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
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Optional, Tuple

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

# Initialize Firebase only once
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.stop()

db = firestore.client()

def get_current_date() -> date:
    return datetime.now(tz).date()

# === FIRESTORE HELPERS ===
def get_user(username: str) -> Optional[Dict]:
    try:
        doc = db.collection("users").document(username).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        st.error(f"Error accessing user data: {e}")
        return None

def validate_login(username: str, password: str) -> bool:
    user = get_user(username)
    if not user:
        return False
    return user.get("password") == password

def update_user_data(username: str, data: Dict) -> None:
    try:
        db.collection("users").document(username).update(data)
    except Exception as e:
        st.error(f"Error updating user data: {e}")

def init_user(username: str, password: str = "password") -> None:
    if not get_user(username):
        try:
            db.collection("users").document(username).set({
                "password": password,
                "last_checkin_date": "",
                "incentives": [],
                "today_goal": TARGET_DAILY,
                "today_notes": "",
                "is_working": False,
                "checkin_time": None,
                "created_at": datetime.now(tz).isoformat()
            })
        except Exception as e:
            st.error(f"Error creating user: {e}")

def add_entry_to_firestore(entry: Dict) -> None:
    try:
        entry["created_at"] = datetime.now(tz).isoformat()
        db.collection("deliveries").add(entry)
    except Exception as e:
        st.error(f"Error saving delivery: {e}")

def load_user_deliveries(username: str) -> pd.DataFrame:
    try:
        docs = db.collection("deliveries").where("username", "==", username).stream()
        data = [doc.to_dict() for doc in docs]
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading deliveries: {e}")
        return pd.DataFrame()

def add_tip_baiter_to_firestore(entry: Dict) -> None:
    try:
        entry["created_at"] = datetime.now(tz).isoformat()
        db.collection("tip_baiters").add(entry)
    except Exception as e:
        st.error(f"Error saving tip baiter: {e}")

def load_user_tip_baiters(username: str) -> pd.DataFrame:
    try:
        docs = db.collection("tip_baiters").where("username", "==", username).stream()
        data = []
        for doc in docs:
            entry = doc.to_dict()
            entry["id"] = doc.id
            data.append(entry)
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading tip baiters: {e}")
        return pd.DataFrame()

def save_incentives(username: str, incentives: List[Dict]) -> None:
    try:
        db.collection("users").document(username).update({"incentives": incentives})
    except Exception as e:
        st.error(f"Error saving incentives: {e}")

# === INCENTIVE MANAGEMENT ===
def manage_incentives(username: str) -> None:
    """Add, edit, and remove incentives for the current user."""
    st.subheader("💰 Incentive Management")
    
    # Load current incentives
    user_data = get_user(username)
    current_incentives = user_data.get("incentives", [])
    
    with st.expander("➕ Add New Incentive", expanded=False):
        with st.form("incentive_form"):
            col1, col2 = st.columns(2)
            with col1:
                incentive_name = st.text_input("Incentive Name", help="E.g. 'Lunch Rush Bonus'")
                start_time = st.time_input("Start Time", value=time(11, 0))
            with col2:
                applies_to = st.multiselect("Applies To", ORDER_TYPES, default=ORDER_TYPES)
                end_time = st.time_input("End Time", value=time(14, 0))
            
            amount = st.number_input("Bonus Amount Per Order ($)", value=5.0, step=0.5, min_value=0.1)
            notes = st.text_area("Notes (Optional)")
            
            if st.form_submit_button("Save Incentive"):
                if not incentive_name:
                    st.error("Incentive name is required!")
                elif start_time >= end_time:
                    st.error("End time must be after start time!")
                elif not applies_to:
                    st.error("Must select at least one order type!")
                else:
                    new_incentive = {
                        "name": incentive_name,
                        "start_time": start_time.strftime("%H:%M"),
                        "end_time": end_time.strftime("%H:%M"),
                        "applies_to": applies_to,
                        "amount": amount,
                        "notes": notes,
                        "active": True
                    }
                    current_incentives.append(new_incentive)
                    save_incentives(username, current_incentives)
                    st.success("Incentive saved!")
                    st.rerun()
    
    st.subheader("📋 Your Active Incentives")
    if not current_incentives:
        st.info("No incentives set up yet")
    else:
        for idx, incentive in enumerate(current_incentives):
            with st.expander(f"{incentive['name']} - ${incentive['amount']} per order"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    **Time:** {incentive['start_time']} to {incentive['end_time']}  
                    **Applies to:** {', '.join(incentive['applies_to'])}  
                    **Notes:** {incentive.get('notes', 'None')}
                    """)
                with col2:
                    if st.button("🗑️ Remove", key=f"del_incentive_{idx}"):
                        del current_incentives[idx]
                        save_incentives(username, current_incentives)
                        st.success("Incentive removed!")
                        st.rerun()

def apply_incentives(order_type: str, order_time: time, base_amount: float, username: str) -> Tuple[float, float, List[Dict]]:
    """Apply any matching incentives to the order."""
    user_data = get_user(username)
    incentives = user_data.get("incentives", [])
    bonus_amount = 0.0
    applied_incentives = []
    
    for incentive in incentives:
        if incentive.get("active", True) and order_type in incentive.get("applies_to", []):
            start_time = datetime.strptime(incentive["start_time"], "%H:%M").time()
            end_time = datetime.strptime(incentive["end_time"], "%H:%M").time()
            
            if start_time <= order_time <= end_time:
                bonus_amount += incentive["amount"]
                applied_incentives.append({
                    "name": incentive["name"],
                    "amount": incentive["amount"]
                })
    
    total_amount = base_amount + bonus_amount
    return total_amount, bonus_amount, applied_incentives

# === ENHANCED OCR PARSING ===
def extract_text_from_image(image_file) -> List[str]:
    """Extract text from an image using OCR."""
    try:
        reader = easyocr.Reader(["en"], gpu=False)
        img_bytes = image_file.read()
        image_file.seek(0)
        return reader.readtext(img_bytes, detail=0)
    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        return []

def parse_screenshot_text(text_list: List[str]) -> Tuple[datetime, float, float, str]:
    """Parse OCR text to extract delivery information."""
    joined = " ".join(text_list).lower()
    
    # Default values
    ts = datetime.now(tz)
    total = 0.0
    ml = 0.0
    order_type = "Delivery"
    
    try:
        # Improved amount detection with multiple patterns
        amount_patterns = [
            r"\$?(\d{1,3}(?:,\d{3})*\.\d{2})",  # $1,234.56 or 1,234.56
            r"\$?(\d+\.\d{2})\b",               # $12.34 or 12.34
            r"\$(\d+)\b",                        # $12 (whole dollar amounts)
            r"total.*?(\d+\.\d{2})"              # Total: 12.34
        ]
        
        for pattern in amount_patterns:
            dollar_matches = re.findall(pattern, joined)
            if dollar_matches:
                amounts = [float(amt.replace(',', '')) for amt in dollar_matches]
                total = max(amounts)
                break
        
        # Miles detection with multiple patterns
        mile_patterns = [
            r"(\d+(?:\.\d)?)\s?mi(?:les)?",     # 1.2 mi or 1.2 miles
            r"distance.*?(\d+(?:\.\d)?)",        # Distance: 1.2
            r"(\d+(?:\.\d)?)\s?miles?\b"        # 1.2 mile or 1.2 miles
        ]
        
        for pattern in mile_patterns:
            miles = re.findall(pattern, joined)
            if miles:
                ml = float(miles[0])
                break
        
        # Enhanced time parsing
        time_patterns = [
            r"\b(\d{1,2}):(\d{2})\s?([ap]m)?\b",  # 12:30 PM or 12:30
            r"\b(\d{1,2})\s?([ap]m)\b",           # 12 PM
            r"time.*?(\d{1,2}):(\d{2})"           # Time: 12:30
        ]
        
        for pattern in time_patterns:
            time_match = re.search(pattern, joined, re.IGNORECASE)
            if time_match:
                groups = time_match.groups()
                if len(groups) >= 2:
                    hour = int(groups[0])
                    minute = int(groups[1]) if len(groups) > 1 and groups[1].isdigit() else 0
                    period = groups[-1].lower() if len(groups) > 2 and groups[-1] in ['am', 'pm'] else None
                    
                    if period:
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
                break
        
        # Enhanced order type detection
        type_keywords = {
            "Shop": ["shop", "s&d", "shopping", "scan", "item", "shopping order"],
            "Pickup": ["pickup", "curbside", "pick up", "store pickup", "pick-up"]
        }
        
        for t, keywords in type_keywords.items():
            if any(kw in joined for kw in keywords):
                order_type = t
                break
    
    except Exception as e:
        st.error(f"Error parsing OCR text: {e}")
    
    return ts, total, ml, order_type

# === AI ANALYTICS ===
def calculate_performance_metrics(df: pd.DataFrame) -> Dict:
    """Calculate various performance metrics from delivery data."""
    metrics = {
        "total_orders": 0,
        "total_earnings": 0.0,
        "epm": 0.0,
        "eph": 0.0,
        "performance": "Unknown",
        "type_distribution": {}
    }
    
    if df.empty:
        return metrics
    
    try:
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
        
        if "epm" in metrics and "eph" in metrics:
            for level, criteria in PERFORMANCE_LEVELS.items():
                if metrics["epm"] >= criteria["min_epm"] and metrics["eph"] >= criteria["min_eph"]:
                    metrics["performance"] = level
                    break
        
        return metrics
    
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return metrics

def predict_earnings(df: pd.DataFrame, target_date: date) -> Optional[float]:
    """Predict earnings for a target date using machine learning."""
    if df.empty or "date" not in df.columns:
        return None
    
    try:
        # Prepare data
        df_daily = df.groupby("date")["order_total"].sum().reset_index()
        df_daily["date_ordinal"] = df_daily["date"].apply(lambda d: d.toordinal())
        df_daily["day_of_week"] = df_daily["date"].apply(lambda d: d.weekday())
        
        if len(df_daily) < 5:
            return None
        
        # Feature engineering
        X = df_daily[["date_ordinal", "day_of_week"]]
        y = df_daily["order_total"]
        
        # Try more sophisticated model
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
        except:
            # Fall back to linear regression if RF fails
            model = LinearRegression().fit(X, y)
        
        # Make prediction
        target_ordinal = target_date.toordinal()
        target_dow = target_date.weekday()
        prediction = model.predict(np.array([[target_ordinal, target_dow]]))[0]
        
        return max(0, prediction)
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# === TIP BAITER TRACKER ===
def tip_baiter_tracker(username: str) -> None:
    """Display and manage tip baiter tracking functionality."""
    st.subheader("🚨 Tip Baiter Tracker")
    
    # Add new tip baiter
    with st.expander("➕ Add New Tip Baiter", expanded=False):
        with st.form("tip_baiter_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name (Required)", help="Customer name or identifier")
            with col2:
                date_baited = st.date_input("Date", value=get_current_date())
            
            address = st.text_input("Address (Optional)", help="Approximate address or location")
            
            col3, col4 = st.columns(2)
            with col3:
                amount_baited = st.number_input("Amount Baited ($)", value=0.0, step=0.01, min_value=0.0)
            with col4:
                rating = st.slider("Severity (1-5)", 1, 5, 3, 
                                 help="How bad was this tip bait? 1=minor, 5=egregious")
            
            notes = st.text_area("Notes", help="Any additional details about this incident")
            
            if st.form_submit_button("Save Tip Baiter"):
                if not name:
                    st.error("Name is required!")
                else:
                    entry = {
                        "name": name.strip(),
                        "address": address.strip() if address else "",
                        "date": date_baited.isoformat(),
                        "amount": float(amount_baited),
                        "rating": rating,
                        "notes": notes.strip(),
                        "username": username,
                        "timestamp": datetime.now(tz).isoformat()
                    }
                    add_tip_baiter_to_firestore(entry)
                    st.success("Tip baiter saved!")
                    st.rerun()
    
    # Display and manage existing tip baiters
    st.subheader("📋 Your Tip Baiters")
    tip_baiters_df = load_user_tip_baiters(username)
    
    if not tip_baiters_df.empty:
        tip_baiters_df["date"] = pd.to_datetime(tip_baiters_df["date"])
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tip Baiters", len(tip_baiters_df))
        with col2:
            st.metric("Total Amount Baited", f"${tip_baiters_df['amount'].sum():.2f}")
        with col3:
            avg_severity = tip_baiters_df['rating'].mean()
            st.metric("Average Severity", f"{avg_severity:.1f}/5")
        
        # Enhanced filtering
        with st.expander("🔍 Filter Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("Search by name or address")
            with col2:
                date_filter = st.selectbox(
                    "Date Range",
                    ["All", "Today", "Yesterday", "Last 7 days", "Last 30 days", "Last 90 days", "Custom"]
                )
            with col3:
                min_severity = st.slider("Minimum Severity", 1, 5, 1)
        
        # Apply filters
        if search_term:
            tip_baiters_df = tip_baiters_df[
                tip_baiters_df["name"].str.contains(search_term, case=False) |
                tip_baiters_df["address"].str.contains(search_term, case=False)
            ]
        
        if date_filter != "All":
            today = get_current_date()
            if date_filter == "Today":
                tip_baiters_df = tip_baiters_df[tip_baiters_df["date"].dt.date == today]
            elif date_filter == "Yesterday":
                yesterday = today - timedelta(days=1)
                tip_baiters_df = tip_baiters_df[tip_baiters_df["date"].dt.date == yesterday]
            else:
                days = 7 if date_filter == "Last 7 days" else 30 if date_filter == "Last 30 days" else 90
                cutoff_date = today - timedelta(days=days)
                tip_baiters_df = tip_baiters_df[tip_baiters_df["date"].dt.date >= cutoff_date]
        
        # Apply severity filter
        tip_baiters_df = tip_baiters_df[tip_baiters_df["rating"] >= min_severity]
        
        # Display in a more organized way
        if not tip_baiters_df.empty:
            tip_baiters_df = tip_baiters_df.sort_values(["date", "rating"], ascending=[False, False])
            
            # Group by date for better organization
            grouped = tip_baiters_df.groupby(tip_baiters_df["date"].dt.date)
            
            for date_val, group in grouped:
                with st.expander(f"📅 {date_val.strftime('%b %d, %Y')} ({len(group)} incidents)"):
                    for _, row in group.iterrows():
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{row['name']}**")
                                st.caption(f"📍 {row['address'] or 'No address'} | 💰 ${row['amount']:.2f} | ⭐ {row['rating']}/5")
                                if row["notes"]:
                                    st.markdown(f"📝 *{row['notes']}*")
                            with col2:
                                if st.button("🗑️", key=f"del_tb_{row['id']}"):
                                    db.collection("tip_baiters").document(row["id"]).delete()
                                    st.success("Tip baiter removed!")
                                    st.rerun()
                            st.divider()
        else:
            st.info("No tip baiters match your filters")
    else:
        st.info("No tip baiters recorded yet")

# === STREAMLIT UI ===
def login_section() -> Optional[str]:
    """Display login interface and return username if successful."""
    st.title("🔐 Spark Tracker Login")
    
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username").strip().lower()
    with col2:
        password = st.text_input("Password", type="password")
    
    if st.button("Login", type="primary"):
        if not username or not password:
            st.error("Both username and password are required")
            return None
        
        init_user(username, password)
        if validate_login(username, password):
            st.session_state.update({
                "logged_in": True,
                "username": username
            })
            st.rerun()
        else:
            st.error("Invalid credentials")
    
    st.markdown("---")
    st.markdown("Don't have an account? Just enter a new username and password to create one.")
    return None

def daily_checkin(username: str) -> bool:
    """Handle daily check-in process."""
    today = get_current_date()
    
    # Load user data with fallback
    user_data = get_user(username) or {}
    last_ci_str = user_data.get("last_checkin_date", "")
    
    # Simple string comparison (no parsing needed)
    if last_ci_str == today.isoformat():
        # Already checked in today
        is_working = user_data.get("is_working", False)
        
        # Store in session state
        st.session_state.daily_checkin = {
            "working": is_working,
            "goal": user_data.get("today_goal", TARGET_DAILY),
            "notes": user_data.get("today_notes", "")
        }
        
        # Allow changing mind if not working
        if not is_working and st.button("🚀 Actually, I want to work today"):
            update_user_data(username, {
                "is_working": True,
                "today_goal": TARGET_DAILY,
                "today_notes": "Changed mind - decided to work"
            })
            st.rerun()
            
        return is_working
    
    # --- New Check-in Flow ---
    st.header("📅 Daily Check‑In")
    working = st.radio("Working today?", ("Yes", "No"), index=0, horizontal=True)
    
    if working == "Yes":
        goal = st.number_input("Today's Goal ($)", value=TARGET_DAILY)
        notes = st.text_area("Notes")
        
        if st.button("Start Tracking"):
            update_user_data(username, {
                "last_checkin_date": today.isoformat(),
                "is_working": True,
                "today_goal": goal,
                "today_notes": notes
            })
            st.session_state.daily_checkin = {
                "working": True,
                "goal": goal,
                "notes": notes
            }
            st.rerun()
    else:
        if st.button("Take the day off"):
            update_user_data(username, {
                "last_checkin_date": today.isoformat(),
                "is_working": False,
                "today_goal": 0,
                "today_notes": "Day off"
            })
            st.session_state.daily_checkin = {
                "working": False,
                "goal": 0,
                "notes": "Day off"
            }
            st.rerun()
    
    st.stop()
    return False

def delivery_entry_form(username: str, today: date) -> None:
    """Display form for entering delivery information."""
    st.subheader("📝 Order Entry")
    
    # OCR functionality
    uploaded = st.file_uploader("Upload screenshot (optional)", 
                               type=["png", "jpg", "jpeg"],
                               help="Upload a screenshot of your delivery summary to auto-fill details")
    
    parsed = None
    if uploaded:
        with st.spinner("Analyzing with AI..."):
            try:
                text_list = extract_text_from_image(uploaded)
                if text_list:
                    ts, total, ml, order_type = parse_screenshot_text(text_list)
                    parsed = {
                        "timestamp": ts, 
                        "order_total": total,
                        "miles": ml, 
                        "order_type": order_type
                    }
                    st.success(f"✅ AI Analysis: ${total:.2f} | {ml:.1f} mi @ {ts.strftime('%I:%M %p')} | Type: {order_type}")
                else:
                    st.warning("No text found in image")
            except Exception as e:
                st.error(f"Failed to analyze image: {e}")
    
    # Entry form
    with st.form("entry_form", clear_on_submit=True):
        if parsed:
            default_time = parsed["timestamp"].time()
            default_date = parsed["timestamp"].date()
            default_type = parsed["order_type"]
            default_total = parsed["order_total"]
            default_miles = parsed["miles"]
        else:
            now = datetime.now(tz)
            default_time = now.time()
            default_date = today
            default_type = "Delivery"
            default_total = 0.0
            default_miles = 0.0
        
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("Date", value=default_date)
        with col2:
            selected_time = st.time_input("Time", value=time(default_time.hour, default_time.minute))
        
        order_type = st.radio("Order Type", ORDER_TYPES, 
                             index=ORDER_TYPES.index(default_type), 
                             horizontal=True)
        
        col3, col4 = st.columns(2)
        with col3:
            base_amount = st.number_input("Base Amount ($)", 
                                       value=default_total, 
                                       step=0.01,
                                       min_value=0.0,
                                       help="Amount before incentives")
        with col4:
            ml = st.number_input("Miles Driven", 
                                value=default_miles if parsed else 0.0, 
                                step=0.1,
                                min_value=0.0)
        
        # Apply incentives
        total_amount = base_amount
        bonus_amount = 0.0
        applied_incentives = []
        
        if st.form_submit_button("Save Delivery", type="primary"):
            try:
                naive_dt = datetime.combine(selected_date, selected_time)
                aware_dt = tz.localize(naive_dt)
                
                # Apply any matching incentives
                total_amount, bonus_amount, applied_incentives = apply_incentives(
                    order_type, selected_time, base_amount, username
                )
                
                entry = {
                    "timestamp": aware_dt.isoformat(),
                    "order_total": float(total_amount),
                    "base_amount": float(base_amount),
                    "bonus_amount": float(bonus_amount),
                    "incentives": applied_incentives,
                    "miles": float(ml),
                    "earnings_per_mile": round(float(total_amount)/float(ml), 2) if ml else 0.0,
                    "hour": selected_time.hour,
                    "username": username,
                    "order_type": order_type
                }
                
                add_entry_to_firestore(entry)
                
                if bonus_amount > 0:
                    incentive_names = ", ".join([i["name"] for i in applied_incentives])
                    st.success(f"✅ Saved {order_type} at {aware_dt.strftime('%I:%M %p')} with ${bonus_amount:.2f} in bonuses ({incentive_names})!")
                else:
                    st.success(f"✅ Saved {order_type} entry at {aware_dt.strftime('%I:%M %p')}!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save entry: {e}")

def display_metrics(username: str, today: date) -> Tuple[float, float]:
    """Display key metrics and return earned amount and goal."""
    df_all = load_user_deliveries(username)
    if not df_all.empty and "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
        df_all = df_all.dropna(subset=["timestamp"])
        df_all["date"] = df_all["timestamp"].dt.date
        today_df = df_all[df_all["date"] == today]
    else:
        today_df = pd.DataFrame()
    
    daily_checkin = st.session_state.get("daily_checkin", {})
    goal = daily_checkin.get("goal", TARGET_DAILY)
    earned = today_df["order_total"].sum() if not today_df.empty else 0.0
    bonus_earned = today_df["bonus_amount"].sum() if "bonus_amount" in today_df.columns else 0.0
    base_earned = today_df["base_amount"].sum() if "base_amount" in today_df.columns else earned
    perc = min(earned / goal * 100, 100) if goal else 0
    
    # Calculate Earnings Per Hour - FIXED LOGIC
    eph = None
    if "start_time" in daily_checkin and not today_df.empty:
        shift_duration = (datetime.now(tz) - daily_checkin["start_time"]).total_seconds() / 3600
        
        # Ensure we don't divide by zero and have reasonable shift duration
        if shift_duration > 0.1:  # At least 6 minutes of work
            eph = earned / shift_duration
        else:
            # If shift just started, use average of last 5 deliveries
            last_5 = df_all.sort_values("timestamp", ascending=False).head(5)
            if not last_5.empty:
                avg_eph = last_5["order_total"].sum() / 5  # Simple average
                eph = avg_eph if avg_eph > 0 else None
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Today's Earnings", 
                 f"${earned:.2f}", 
                 f"{perc:.0f}% of ${goal} goal",
                 delta_color="off" if perc >= 100 else "inverse")
        if bonus_earned > 0:
            st.caption(f"${base_earned:.2f} base + ${bonus_earned:.2f} bonuses")
    with col2:
        st.metric("Orders Completed", 
                 len(today_df) if not today_df.empty else 0)
    with col3:
        if eph is not None:
            # Cap EPH at reasonable maximum (e.g., $100/hr) to prevent unrealistic spikes
            realistic_eph = min(eph, 100)
            st.metric("Earnings Per Hour", 
                     f"${realistic_eph:.2f}",
                     "good" if realistic_eph >= 25 else "normal" if realistic_eph >= 20 else "bad")
        else:
            st.metric("Earnings Per Hour", "Calculating...")
    with col4:
        if not today_df.empty and "miles" in today_df.columns and today_df["miles"].sum() > 0:
            epm = earned / today_df["miles"].sum()
            st.metric("Earnings Per Mile", 
                     f"${epm:.2f}",
                     "good" if epm >= 2.5 else "normal" if epm >= 1.5 else "bad")
    
    return earned, goal

def display_ai_insights(username: str, today: date) -> None:
    """Display AI-generated insights and recommendations."""
    st.subheader("🧠 AI Insights")
    
    df_all = load_user_deliveries(username)
    if not df_all.empty and "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
        df_all = df_all.dropna(subset=["timestamp"])
        df_all["date"] = df_all["timestamp"].dt.date
        today_df = df_all[df_all["date"] == today]
    else:
        today_df = pd.DataFrame()
    
    # Performance Rating
    metrics = calculate_performance_metrics(today_df) if not today_df.empty else {}
    
    if "performance" in metrics:
        performance = metrics["performance"]
        color = {
            "Excellent": "green",
            "Good": "blue", 
            "Fair": "orange", 
            "Poor": "red"
        }.get(performance, "gray")
        
        st.markdown(f"### Performance Rating: :{color}[{performance}]")
        
        if "eph" in metrics and "epm" in metrics:
            st.markdown(f"- Earnings Per Hour: ${metrics['eph']:.2f}")
            st.markdown(f"- Earnings Per Mile: ${metrics['epm']:.2f}")
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    if not today_df.empty:
        if "eph" in metrics and metrics["eph"] < 20:
            st.warning("""
            **Try working during busier hours** to increase your earnings per hour. 
            Consider focusing on meal times (11am-2pm, 5pm-8pm) when order volume is higher.
            """)
        elif "epm" in metrics and metrics["epm"] < 2.0:
            st.warning("""
            **Focus on orders with shorter distances** to improve earnings per mile. 
            Look for orders that pay at least $2 per mile.
            """)
        else:
            st.success("""
            **You're doing great!** Keep up the good work. 
            Consider tracking your performance over time to identify your most profitable patterns.
            """)
        
        # Time-based recommendations
        if "hour" in today_df.columns:
            hourly_earnings = today_df.groupby("hour")["order_total"].sum()
            if not hourly_earnings.empty:
                best_hour = hourly_earnings.idxmax()
                worst_hour = hourly_earnings.idxmin()
                
                st.info(f"""
                **Best hour today**: {best_hour}:00 - ${hourly_earnings[best_hour]:.2f}  
                **Worst hour today**: {worst_hour}:00 - ${hourly_earnings[worst_hour]:.2f}
                """)
    else:
        st.info("Complete a few deliveries to unlock personalized recommendations")

def display_analytics(username: str) -> None:
    """Display analytics dashboard with interactive charts."""
    st.subheader("📊 Analytics Dashboard")
    
    df_all = load_user_deliveries(username)
    if df_all.empty or "timestamp" not in df_all.columns:
        st.info("Complete a few deliveries to unlock analytics")
        return
    
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
    df_all["date"] = df_all["timestamp"].dt.date
    df_all["hour"] = df_all["timestamp"].dt.hour
    df_all["hour_12"] = df_all["timestamp"].dt.strftime("%I %p")
    df_all["day_of_week"] = df_all["timestamp"].dt.day_name()
    df_all["week"] = df_all["timestamp"].dt.isocalendar().week
    
    # Tabbed interface for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Order Types", "Hourly", "Weekly"])
    
    with tab1:
        st.write("### 📅 Earnings Trends")
        
        col1, col2 = st.columns(2)
        with col1:
            time_period = st.selectbox("Time Period", 
                                     ["Daily", "Weekly", "Monthly"],
                                     key="trend_period")
        with col2:
            metric = st.selectbox("Metric",
                                ["Total Earnings", "Average per Order", "Number of Orders"],
                                key="trend_metric")
        
        # Prepare data based on selections
        if time_period == "Daily":
            group_col = "date"
            x_title = "Date"
        elif time_period == "Weekly":
            group_col = "week"
            x_title = "Week"
        else:  # Monthly
            df_all["month"] = df_all["timestamp"].dt.to_period("M").astype(str)
            group_col = "month"
            x_title = "Month"
        
        if metric == "Total Earnings":
            y_col = "order_total"
            y_title = "Total Earnings ($)"
            agg_func = "sum"
        elif metric == "Average per Order":
            y_col = "order_total"
            y_title = "Average per Order ($)"
            agg_func = "mean"
        else:  # Number of Orders
            y_col = "order_total"
            y_title = "Number of Orders"
            agg_func = "count"
        
        trend_data = df_all.groupby(group_col)[y_col].agg(agg_func).reset_index()
        
        fig = px.line(trend_data, x=group_col, y=y_col, markers=True,
                     title=f"{metric} by {time_period}")
        fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.write("### 📦 Order Type Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            type_counts = df_all["order_type"].value_counts().reset_index()
            fig1 = px.pie(type_counts, values="count", names="order_type", 
                         hole=0.4, title="Order Type Distribution")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            type_earnings = df_all.groupby("order_type")["order_total"].sum().reset_index()
            fig2 = px.bar(type_earnings, x="order_type", y="order_total", 
                         color="order_type", title="Total Earnings by Order Type")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Earnings per mile by order type
        if "miles" in df_all.columns:
            type_epm = df_all.groupby("order_type").apply(
                lambda x: x["order_total"].sum() / x["miles"].sum() if x["miles"].sum() > 0 else 0
            ).reset_index(name="epm")
            
            fig3 = px.bar(type_epm, x="order_type", y="epm", 
                         color="order_type", title="Earnings Per Mile by Order Type")
            fig3.update_yaxes(title_text="Earnings Per Mile ($)")
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.write("### ⏰ Hourly Performance")
        
        hourly_data = df_all.groupby("hour").agg({
            "order_total": ["mean", "sum", "count"],
            "miles": "mean"
        }).reset_index()
        
        # Flatten multi-index columns
        hourly_data.columns = ['_'.join(col).strip() for col in hourly_data.columns.values]
        hourly_data = hourly_data.rename(columns={
            "hour_": "hour",
            "order_total_mean": "avg_earnings",
            "order_total_sum": "total_earnings",
            "order_total_count": "order_count",
            "miles_mean": "avg_miles"
        })
        
        # Add formatted hour labels
        hourly_data["hour_str"] = hourly_data["hour"].apply(
            lambda h: f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}")
        
        metric = st.selectbox("Hourly Metric", 
                            ["Average Earnings", "Total Earnings", "Order Count"],
                            key="hourly_metric")
        
        if metric == "Average Earnings":
            y_col = "avg_earnings"
            title = "Average Earnings by Hour"
            y_title = "Average Earnings ($)"
        elif metric == "Total Earnings":
            y_col = "total_earnings"
            title = "Total Earnings by Hour"
            y_title = "Total Earnings ($)"
        else:
            y_col = "order_count"
            title = "Order Count by Hour"
            y_title = "Number of Orders"
        
        fig = px.bar(hourly_data, x="hour_str", y=y_col, title=title)
        fig.update_layout(xaxis_title="Hour", yaxis_title=y_title)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.write("### 📆 Weekly Performance Heatmap")
        
        heat_data = df_all.groupby(["day_of_week", "hour"])["order_total"].mean().unstack()
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat_data = heat_data.reindex(days_order)
        
        fig = px.imshow(heat_data, 
                        labels=dict(x="Hour", y="Day", color="Average Earnings"),
                        x=[f"{h}:00" for h in heat_data.columns],
                        y=heat_data.index,
                        color_continuous_scale="Viridis",
                        title="Average Earnings by Day and Hour")
        st.plotly_chart(fig, use_container_width=True)

def delete_entries_section(username: str) -> None:
    """Display interface for deleting entries."""
    st.subheader("🗑️ Delete Entries")
    selected_date = st.date_input("Select date to manage entries", 
                                value=get_current_date(), 
                                key="delete_date")
    
    df_all = load_user_deliveries(username)
    if not df_all.empty and "timestamp" in df_all.columns:
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
        df_all["date"] = df_all["timestamp"].dt.date
        entries_to_show = df_all[df_all["date"] == selected_date]
    else:
        entries_to_show = pd.DataFrame()

    if not entries_to_show.empty:
        entries_to_show = entries_to_show.sort_values("timestamp")
        
        for _, row in entries_to_show.iterrows():
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    **🕒 {row['timestamp'].strftime('%I:%M %p')}**  
                    💵 ${row['order_total']:.2f} ({row['order_type']})  
                    🚗 {row.get('miles', 0):.1f} mi (EPM: ${row.get('earnings_per_mile', 0):.2f})
                    """)
                    if row.get("bonus_amount", 0) > 0:
                        st.caption(f"✨ ${row['bonus_amount']:.2f} in bonuses")
                with col2:
                    if st.button("Delete", key=f"del_{row.name}"):
                        # Find the exact document to delete
                        docs = db.collection("deliveries").where("username", "==", username)\
                                                         .where("timestamp", "==", row["timestamp"].isoformat())\
                                                         .where("order_total", "==", row["order_total"])\
                                                         .limit(1).stream()
                        
                        for doc in docs:
                            db.collection("deliveries").document(doc.id).delete()
                            st.success("Entry deleted!")
                            st.rerun()
                st.divider()
    else:
        st.info("No entries found for this date")

# === MAIN APP FLOW ===
if "logged_in" not in st.session_state:
    login_section()
    st.stop()

user = st.session_state["username"]
today = get_current_date()

# Check if user has checked in today
if daily_checkin(user):
    # Main interface
    st.title("📦 Spark Delivery Tracker")
    
    # Display daily notes if available
    daily_notes = st.session_state.get("daily_checkin", {}).get("notes", "")
    if daily_notes:
        with st.expander("📝 Today's Notes"):
            st.write(daily_notes)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tracker", "Incentives", "Tip Baiters", "Analytics", "Settings"])
    
    with tab1:
        # Display current metrics
        earned, goal = display_metrics(user, today)
        
        # Delivery entry form
        delivery_entry_form(user, today)
        
        # AI Insights
        display_ai_insights(user, today)
    
    with tab2:
        manage_incentives(user)
    
    with tab3:
        tip_baiter_tracker(user)
    
    with tab4:
        display_analytics(user)
    
    with tab5:
        delete_entries_section(user)
    
    st.caption("🧠 AI-Powered Spark Tracker v3.0 | Data stays 100% yours.")
else:
    st.success("🏝️ Enjoy your day off!")
    st.stop()
