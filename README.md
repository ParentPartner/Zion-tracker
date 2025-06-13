
# 🚦 Spark Delivery Tracker (v2.0)

A **Streamlit**-based delivery earnings tracker and analytics dashboard with AI‑enhanced OCR parsing and a tip‑baiter tracker. Save delivery records via screenshot or manual entry, track daily performance, automatically predict earnings, and visualize trends.

---

## 🧠 Features

- **User Authentication & Daily Check‑In**  
  - Simple credential validation via Firebase (Firestore).  
  - Daily “working or day-off” check-in with goal setting and motivational notes.

- **OCR-Powered Order Parsing**  
  - Upload screenshots (PNG, JPG, JPEG) for AI-powered extraction of timestamp, total, distance, and order type.

- **Order Entry Form**  
  - Use parsed data as defaults or manually enter Date, Time, Type (Delivery, Shop, Pickup), Total, Miles.

- **Performance Tracking**  
  - Real-time metrics: Today’s earnings, % of goal, earnings/hour, earnings/mile.  
  - AI insights including performance rating (Excellent/Good/Fair/Poor) and tomorrow's earnings prediction with `LinearRegression`.

- **Tip Baiter Tracker**  
  - Log and manage “tip baiters” (names, dates, amounts, notes).  
  - View totals and recent entries; filterable/searchable.

- **Analytics Dashboard**  
  - Visualizations with Plotly:  
    - Daily Earnings Trend (line chart)  
    - Order Type Breakdown (pie chart + bar chart)  
    - Hourly Earnings distribution  
    - Weekly Heatmap (average earnings by day and hour)

- **Entry Management**  
  - Delete specific entries by date directly from the interface.

---

## 💾 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io)  
- **Data Storage**: [Firebase Firestore](https://firebase.google.com/)  
- **AI/OCR**: [EasyOCR](https://github.com/JaidedAI/EasyOCR), plus `pytz`, `re`  
- **Analytics**: `pandas`, `numpy`, `scikit-learn` (LinearRegression)  
- **Visualization**: [Plotly Express](https://plotly.com/python/plotly-express)

---

## 📥 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/ParentPartner/Zion-tracker.git
   cd Zion-tracker

	2.	Install dependencies

pip install -r requirements.txt


	3.	Set up Firebase credentials
	•	Add your Firebase service account JSON in streamlit.secrets under "firebase".
	•	Example ~/.streamlit/secrets.toml:

[firebase]
project_id = "your_project_id"
private_key_id = "your_private_key_id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your_service_account_email"


	•	Or define the secrets as environment variable: ST_SECRETS.

	4.	Launch the app

streamlit run spark_tracker.py



⸻

⚙️ Usage Workflow
	1.	Login or first-time registration (default password is "password").
	2.	Daily check‑in: mark working status, set your goal, take notes.
	3.	Upload screenshots of orders for automatic parsing.
	4.	Manually adjust or enter orders: date, time, type, earnings, miles.
	5.	Track performance in real time on dashboard metrics.
	6.	Inspect trends and visualizations in the Analytics tab.
	7.	Add or manage tip baiters.
	8.	Delete incorrect entries via date selector.

⸻

🛠️ Configuration
	•	Time zone: Default is US/Eastern (tz = pytz.timezone("US/Eastern"))
	•	Target daily earnings: Default is $200, can be modified
	•	Performance levels: Edit PERFORMANCE_LEVELS in the script to adjust thresholds for hourly and per-mile rates

⸻

🔐 Firebase Collections
	•	users: { username, password, last_checkin_date }
	•	deliveries: { timestamp, earnings, miles, order_type, username }
	•	tip_baiters: { name, date, amount, note, username }

⸻

🚀 Extending the App
	•	Add user registration, password reset, or social login
	•	Enhance OCR parsing for additional formats or languages
	•	Build predictive models using weekdays, traffic, weather
	•	Add map visualization using folium or Plotly geospatial
	•	Export data to CSV or sync with other platforms

⸻

💡 Acknowledgments
	•	EasyOCR for AI-powered text extraction
	•	Plotly for interactive visual charts
	•	Streamlit community for UI components and state handling patterns

⸻

📄 License

Distributed under the MIT License. See LICENSE for more details.

⸻

📜 Legal Disclaimer

This tool is intended for personal use only by delivery drivers to track their own performance and insights.
	•	We are not affiliated with, endorsed by, or connected to Walmart, Spark Driver, or any related delivery platform.
	•	No user data is collected or transmitted to third parties beyond your personal Firebase Firestore account.
	•	Use this tool at your own discretion. The authors of this application assume no liability for how this software is used or for any decisions made based on its output.

If you have concerns about terms of service, privacy, or data usage, please consult the official policies of the delivery platform you use.

⸻

🤝 Connect

For 🐞 bug reports, feature suggestions, or collaboration:
	•	Open an issue or PR on the GitHub repository

⸻
