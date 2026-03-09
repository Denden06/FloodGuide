from flask import Flask, render_template, jsonify, request
import requests, joblib, pandas as pd
from datetime import datetime
import mysql.connector, os

app = Flask(__name__)

# =============================
# DATABASE CONNECTION
# =============================
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQLHOST","caboose.proxy.rlwy.net"),
        user=os.getenv("MYSQLUSER","root"),
        password=os.getenv("MYSQLPASSWORD","cYHockIgVucbKAkgNvkRjTqsTGngkjvD"),
        database=os.getenv("MYSQLDATABASE","railway"),
        port=int(os.getenv("MYSQLPORT",52603))
    )

# =============================
# CONFIG
# =============================
API_KEY = "e2a8ee9c6e8ec237763497022a1309bb"
clf = joblib.load("model_class.pkl")

latest_sensor_data = {"water_level":0.0,"rainfall":0.0,"flow_rate":0.0}

MONITORED_SITES = [
    {"id":"bridge1","name":"Mandaue-Mactan Bridge 1","lat":10.326490,"lng":123.952142},
    {"id":"bridge2","name":"Marcelo Fernan Bridge","lat":10.334968,"lng":123.960462}
]

# =============================
# ESP32 SENSOR ENDPOINT
# =============================
@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor():
    data = request.get_json()
    if not data:
        return jsonify({"error":"No JSON received"}),400

    latest_sensor_data["water_level"] = float(data.get("waterLevel",0))
    latest_sensor_data["rainfall"] = float(data.get("totalRain",0))
    latest_sensor_data["flow_rate"] = float(data.get("flowRate",0))

    # Save to DB
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "INSERT INTO sensor_readings (total_rain, water_level, flow_rate) VALUES (%s,%s,%s)"
        cursor.execute(query,(latest_sensor_data["rainfall"],latest_sensor_data["water_level"],latest_sensor_data["flow_rate"]))
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Data saved to database")
    except Exception as e:
        print("❌ Database Insert Error:", e)

    print("Received from ESP32:", data)
    return jsonify({"status":"received"})

# =============================
# MAP DATA ENDPOINT
# =============================
@app.route("/api/map-data")
def map_data():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_list = []

    for loc in MONITORED_SITES:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={loc['lat']}&lon={loc['lng']}&appid={API_KEY}&units=metric"
            r = requests.get(url,timeout=5).json()
            rainfall = r.get("rain",{}).get("1h",0)
            temp = r["main"]["temp"]
            humidity = r["main"]["humidity"]
        except:
            rainfall,temp,humidity = 0,28,70

        water_level_cm = latest_sensor_data["water_level"]

        # ML prediction
        df = pd.DataFrame([{"Rainfall(mm)":rainfall,"Temperature(C)":temp,"Humidity(%)":humidity}])
        pred = clf.predict(df)[0]
        conf = round(max(clf.predict_proba(df)[0])*100,2)
        color = "green" if pred==0 else "orange" if pred==1 else "red"
        risk = "Low" if pred==0 else "Moderate" if pred==1 else "High"

        popup = f"""
        <div style="font-size:14px;">
            <b>{loc['name']}</b><br><br>
            🌊 <b>Flood Risk:</b> <span style="color:{color}; font-weight:bold;">{risk}</span><br>
            🎯 <b>Confidence:</b> {conf}%<br><br>
            🌧 Rainfall: {rainfall:.2f} mm<br>
            📏 Water Level: {water_level_cm:.2f} cm<br>
            🌡 Temperature: {temp:.1f} °C<br>
            💧 Humidity: {humidity}%<br><br>
            🕒 Updated: {timestamp}
        </div>
        """
        data_list.append({"id":loc["id"],"lat":loc["lat"],"lng":loc["lng"],"risk_color":color,"popup_html":popup})

    return jsonify({"locations":data_list})

# =============================
# MAIN PAGE
# =============================
@app.route("/")
def home():
    return render_template("map.html")

if __name__ == "__main__":
    print("FloodGuide Running...")
    app.run(host="0.0.0.0", port=5000, debug=True)