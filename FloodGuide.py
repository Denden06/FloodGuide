from flask import Flask, render_template, jsonify, request
import requests
import joblib
import pandas as pd
from datetime import datetime
import mysql.connector
import os

app = Flask(__name__)

# =========================================
# DATABASE CONNECTION (Railway / Render)
# =========================================
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("mysql.railway.internal", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("cYHockIgVucbKAkgNvkRjTqsTGngkjvD", ""),
        database=os.getenv("DB_NAME", "railway"),
        port=int(os.getenv("DB_PORT", 3306))
    )

# =========================================
# CONFIG
# =========================================
API_KEY = "e2a8ee9c6e8ec237763497022a1309bb"

clf = joblib.load("model_class.pkl")

latest_sensor_data = {
    "water_level": 0.0,
    "rainfall": 0.0
}

MONITORED_SITES = [
    {"id": "bridge1", "name": "Mandaue-Mactan Bridge 1", "lat": 10.326490, "lng": 123.952142},
    {"id": "bridge2", "name": "Marcelo Fernan Bridge", "lat": 10.334968, "lng": 123.960462}
]

# =========================================
# ESP32 SENSOR ENDPOINT
# =========================================
@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor():

    data = request.get_json()
    print("Received from ESP32:", data)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    water_level = float(data.get("waterLevel", 0))
    rainfall = float(data.get("totalRain", 0))

    latest_sensor_data["water_level"] = water_level
    latest_sensor_data["rainfall"] = rainfall

    # 🔥 SAVE TO DATABASE
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO sensor_readings (total_rain, water_level, flow_rate)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (rainfall, water_level, 0))

        conn.commit()
        cursor.close()
        conn.close()

        print("Data saved to database")

    except Exception as e:
        print("Database Insert Error:", e)

    return jsonify({"status": "received"})


# =========================================
# MACHINE LEARNING
# =========================================
def ml_predict(rain, temp, humidity):

    df = pd.DataFrame([{
        "Rainfall(mm)": rain,
        "Temperature(C)": temp,
        "Humidity(%)": humidity
    }])

    pred = clf.predict(df)[0]
    probabilities = clf.predict_proba(df)[0]
    confidence = round(max(probabilities) * 100, 2)

    if pred == 0:
        return "Low", "green", confidence
    elif pred == 1:
        return "Moderate", "orange", confidence
    else:
        return "High", "red", confidence


# =========================================
# MAP DATA ENDPOINT
# =========================================
@app.route("/api/map-data")
def map_data():

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_list = []

    for loc in MONITORED_SITES:

        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={loc['lat']}&lon={loc['lng']}&appid={API_KEY}&units=metric"
            r = requests.get(url, timeout=5).json()

            rainfall = r.get("rain", {}).get("1h", 0)
            temp = r["main"]["temp"]
            humidity = r["main"]["humidity"]

        except:
            rainfall = 0
            temp = 28
            humidity = 70

        water_level_cm = latest_sensor_data["water_level"]

        risk, color, confidence = ml_predict(rainfall, temp, humidity)

        popup = f"""
        <div style="font-size:14px;">
            <b>{loc['name']}</b><br><br>
            🌊 <b>Flood Risk:</b> <span style="color:{color}; font-weight:bold;">{risk}</span><br>
            🎯 <b>Confidence:</b> {confidence}%<br><br>
            🌧 Rainfall: {rainfall:.2f} mm<br>
            📏 Water Level: {water_level_cm:.2f} cm<br>
            🌡 Temperature: {temp:.1f} °C<br>
            💧 Humidity: {humidity}%<br><br>
            🕒 Updated: {timestamp}
        </div>
        """

        data_list.append({
            "id": loc["id"],
            "lat": loc["lat"],
            "lng": loc["lng"],
            "risk_color": color,
            "popup_html": popup
        })

    return jsonify({"locations": data_list})


# =========================================
# MAIN PAGE
# =========================================
@app.route("/")
def home():
    return render_template("map.html")


# =========================================
# RUN LOCAL (Render uses gunicorn)
# =========================================
if __name__ == "__main__":
    print("FloodGuide Running...")
    app.run(host="0.0.0.0", port=5000, debug=True)