from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
from datetime import datetime
import mysql.connector
import joblib
import os

app = Flask(__name__)

# =========================================
# DATABASE CONNECTION (Railway / Render)
# =========================================
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQLHOST", "caboose.proxy.rlwy.net"),
        user=os.getenv("MYSQLUSER", "root"),
        password=os.getenv("MYSQLPASSWORD", "cYHockIgVucbKAkgNvkRjTqsTGngkjvD"),
        database=os.getenv("MYSQLDATABASE", "railway"),
        port=int(os.getenv("MYSQLPORT", 52603))
    )

# =========================================
# CONFIG
# =========================================
API_KEY = "e2a8ee9c6e8ec237763497022a1309bb"
MODEL_PATH = "model_class.pkl"

latest_sensor_data = {
    "water_level": 0.0,
    "rainfall": 0.0,
    "flow_rate": 0.0
}

MONITORED_SITES = [
    {"id": "bridge1", "name": "Mandaue-Mactan Bridge 1", "lat": 10.326490, "lng": 123.952142},
    {"id": "bridge2", "name": "Marcelo Fernan Bridge", "lat": 10.334968, "lng": 123.960462}
]

# =========================================
# LOAD MODEL FUNCTION (auto-reload)
# =========================================
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            clf = joblib.load(MODEL_PATH)
            return clf
        except Exception as e:
            print("❌ Error loading model:", e)
            return None
    else:
        print("❌ model_class.pkl not found!")
        return None

# =========================================
# PREDICTION FUNCTION (matches trained features)
# =========================================
def ml_predict_from_sensor(sensor_data, clf):
    # Build the DataFrame using the exact column names the model expects
    df = pd.DataFrame([{
        "total_rain": sensor_data["rainfall"],   # match training
        "water_level": sensor_data["water_level"],
        "flow_rate": sensor_data["flow_rate"],
        "rain_3h": sensor_data.get("rain_3h", sensor_data["rainfall"]),  # placeholder
        "rain_6h": sensor_data.get("rain_6h", sensor_data["rainfall"]),  # placeholder
        "rise_rate": 0,    # placeholder
        "month": datetime.now().month
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
# ESP32 SENSOR ENDPOINT
# =========================================
@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    water_level = float(data.get("waterLevel", 0))
    rainfall = float(data.get("totalRain", 0))
    flow_rate = float(data.get("flowRate", 0))

    latest_sensor_data["water_level"] = water_level
    latest_sensor_data["rainfall"] = rainfall
    latest_sensor_data["flow_rate"] = flow_rate

    # Save to DB
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "INSERT INTO sensor_readings (total_rain, water_level, flow_rate) VALUES (%s,%s,%s)"
        cursor.execute(query, (rainfall, water_level, flow_rate))
        conn.commit()
        print(f"✅ DB Insert Success: rain={rainfall}, water={water_level}, flow={flow_rate}")
    except mysql.connector.Error as e:
        print(f"❌ MySQL Error: {e}")
    except Exception as e:
        print(f"❌ Unknown DB Error: {e}")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

    print("Received from ESP32:", data)
    return jsonify({"status": "received"})

# =========================================
# MAP DATA ENDPOINT
# =========================================
@app.route("/api/map-data")
def map_data():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_list = []

    clf = load_model()
    if clf is None:
        return jsonify({"error": "Model not loaded"}), 500

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

        # Predict flood risk
        risk, color, confidence = ml_predict_from_sensor(latest_sensor_data, clf)

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
# RUN APP
# =========================================
if __name__ == "__main__":
    print("FloodGuide Running...")
    app.run(host="0.0.0.0", port=5000, debug=True)
@app.route("/api/retrain", methods=["POST"])
def retrain():
    from train_model import load_sensor_data, prepare_features, train_models

    df = load_sensor_data()
    X, y_class, y_reg = prepare_features(df)

    if X is not None:
        train_models(X, y_class, y_reg)
        return {"status": "retrained"}
    else:
        return {"status": "no data"}, 400