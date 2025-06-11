# 🚗 Spark Delivery Tracker

Track your Spark (Walmart) delivery earnings with a smart, easy-to-use Streamlit app. Upload screenshots to auto-extract pay and mileage using OCR, get smart hourly suggestions, and store everything in Google Sheets or locally.

> 🔒 Simple login protected. No data is sent to anyone other than **your** Google Sheet.

---

## 📌 Features

- ✅ Auto-extracts delivery pay and miles from screenshots  
- 🕒 Detects timestamp from screenshot or uses manual input  
- 📈 Visualizes daily totals and hourly average earnings  
- 📊 Calculates $ per mile and averages  
- 🧠 Suggests best hour based on historical performance  
- 📤 Syncs entries to Google Sheets (or local CSV fallback)  

---

## 📸 How It Works

Upload a screenshot from the Spark app and the tool will extract:
- **Order Total** — looks for "estimate" or the largest dollar amount  
- **Miles** — finds mileage like `2.5 mi`  
- **Time** — pulls from screenshot time or lets you pick manually  

> You can edit or override any values manually before saving.

---

## 🛠️ Tech Stack

| Purpose       | Technology                              |
|--------------|----------------------------------------|
| App UI       | [Streamlit](https://streamlit.io)      |
| OCR          | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| Data Handling | `pandas`, CSV                         |
| Cloud Sync   | Google Sheets API                      |
| Auth         | Streamlit Secrets                      |
| Visualization | Streamlit built-in charts             |

---


⚠️ Developer Note

🧠 This app is intended for technical users or developers comfortable working with Python, Streamlit, API credentials, and setting up .toml files.

If you’re not familiar with these concepts, we recommend waiting for the public-friendly version, which will be released later with a cleaner setup, no need for code edits, and safer handling of credentials.

⸻

🤖 About the Code

Parts of this project — including logic, layout, and automation — were assisted or generated using AI tools (e.g., OpenAI’s ChatGPT).

The goal is to rapidly prototype and iterate, but human oversight, testing, and refinement are still essential. Contributions and improvements are welcome!

⸻

⚠️ Legal Disclaimer

This project is an independent tool built for personal use and data tracking.
	•	❌ Not affiliated with, endorsed by, or supported by Spark, Walmart, or any related entities.
	•	❌ Not a cheat, hack, automation tool, or system designed to manipulate Spark’s operations in any way.
	•	✅ Only a personal finance and productivity tool to help drivers log and understand their performance.

Use at your own discretion and ensure compliance with all terms of service.

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ParentPartner/spark-tracker.git
cd spark-tracker
