from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
from datetime import datetime
import mysql.connector
import joblib
import os
import threading
import time
from werkzeug.exceptions import HTTPException

app = Flask(__name__)


def api_error_response(message, status_code=500, error_type="error"):
    return jsonify({
        "status": "error",
        "error": message,
        "type": error_type,
        "status_code": status_code,
    }), status_code


@app.errorhandler(404)
def handle_404(error):
    if request.path.startswith("/api/"):
        return api_error_response(f"API route not found: {request.path}", 404, "not_found")
    return error


@app.errorhandler(500)
def handle_500(error):
    if request.path.startswith("/api/"):
        return api_error_response("Internal server error while serving API request.", 500, "server_error")
    return error


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if request.path.startswith("/api/"):
        if isinstance(error, HTTPException):
            return api_error_response(error.description, error.code or 500, "http_error")
        print(f"❌ Unhandled API error on {request.path}: {error}")
        return api_error_response(str(error), 500, "unhandled_exception")
    raise error

# =========================================
# DATABASE CONNECTION (Clever Cloud)
# =========================================
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQLHOST", "b7hubkhd0btnnd1gtw6f-mysql.services.clever-cloud.com"),
        user=os.getenv("MYSQLUSER", "ufmtd04ysstwmoml"),
        password=os.getenv("MYSQLPASSWORD", "X0kiVkBuBLacKOxNtUkI"),
        database=os.getenv("MYSQLDATABASE", "b7hubkhd0btnnd1gtw6f"),
        port=int(os.getenv("MYSQLPORT", 3306)),
        connection_timeout=10,
        ssl_disabled=False
    )

# =========================================
# CONFIG
# =========================================
API_KEY    = "e2a8ee9c6e8ec237763497022a1309bb"
MODEL_PATH = "model_class.pkl"
PAGASA_DATASET_PATH = os.getenv("PAGASA_DATASET_PATH", "pagasa_labeled.csv")
BOOTSTRAP_TABLE = "bootstrap_training_data"
PREDICTION_AUDIT_TABLE = "prediction_audit"
READING_INTERVAL_MINUTES = 10
RAIN_3H_WINDOW_READINGS = max(1, int(180 / READING_INTERVAL_MINUTES))
RISE_RATE_WINDOW_READINGS = 1
GOOGLE_MAPS_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY",
    "AIzaSyBfyF6DofStbAoUKcG83eEBE72OpaLqTus"
)

# Separate in-memory state for each bridge
latest_sensor_data = {
    "bridge1": {
        "water_level": 0.0,
        "rainfall":    0.0,
        "flow_rate":   0.0,
        "rain_3h":     0.0,
        "rise_rate":   0.0
    },
    "bridge2": {
        "water_level": 0.0,
        "rainfall":    0.0,
        "flow_rate":   0.0,
        "rain_3h":     0.0,
        "rise_rate":   0.0
    }
}

SIMULATION_BRIDGE_IDS = ("bridge1", "bridge2")
prediction_audit_table_ready = False

MONITORED_SITES = [
    {"id": "bridge1", "name": "Mandaue-Mactan Bridge 1", "lat": 10.326490, "lng": 123.952142},
    {"id": "bridge2", "name": "Marcelo Fernan Bridge",   "lat": 10.334968, "lng": 123.960462}
]

# =========================================
# RENDER KEEP-ALIVE
# =========================================
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "")

def keep_alive():
    while True:
        time.sleep(600)
        if RENDER_URL:
            try:
                requests.get(f"{RENDER_URL}/ping", timeout=10)
                print("✅ Keep-alive ping sent")
            except Exception as e:
                print(f"⚠️ Keep-alive ping failed: {e}")

@app.route("/ping")
def ping():
    return jsonify({"status": "alive", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# =========================================
# LOAD MODEL
# =========================================
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            clf = joblib.load(MODEL_PATH)
            if hasattr(clf, 'classes_') and len(clf.classes_) < 2:
                print("⚠️ Model only has 1 class — using rule-based fallback.")
                return None
            return clf
        except Exception as e:
            print("❌ Error loading model:", e)
            return None
    else:
        print("❌ model_class.pkl not found!")
        return None


def prepare_feature_frame(df):
    if df is None or df.empty:
        return df

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])

    sort_cols = ["timestamp"]
    if "device_id" in work.columns:
        sort_cols = ["device_id", "timestamp"]
    if "id" in work.columns:
        sort_cols.append("id")

    work = work.sort_values(sort_cols).reset_index(drop=True)

    if "device_id" in work.columns:
        work["rain_3h"] = work.groupby("device_id")["total_rain"].transform(
            lambda s: s.rolling(window=RAIN_3H_WINDOW_READINGS, min_periods=1).sum()
        )
        work["rise_rate"] = work.groupby("device_id")["water_level"].diff().fillna(0)
    else:
        work["rain_3h"] = work["total_rain"].rolling(window=RAIN_3H_WINDOW_READINGS, min_periods=1).sum()
        work["rise_rate"] = work["water_level"].diff().fillna(0)

    work["month"] = work["timestamp"].dt.month

    final_sort = ["timestamp"]
    if "id" in work.columns:
        final_sort.append("id")

    return work.sort_values(final_sort).reset_index(drop=True).fillna(0)


def normalize_flood_label(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    label_map = {
        "0": 0, "low": 0, "low risk": 0,
        "1": 1, "moderate": 1, "moderate risk": 1, "medium": 1,
        "2": 2, "high": 2, "high risk": 2,
    }
    if text in label_map:
        return label_map[text]
    try:
        numeric = int(float(text))
        return numeric if numeric in (0, 1, 2) else None
    except Exception:
        return None


def flood_label_text(value):
    mapping = {0: "Low", 1: "Moderate", 2: "High"}
    return mapping.get(int(value), "Low")


def prediction_bucket_for(ts, minutes=10):
    bucket_minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=bucket_minute, second=0, microsecond=0)


def load_pagasa_dataset():
    if not PAGASA_DATASET_PATH or not os.path.exists(PAGASA_DATASET_PATH):
        return pd.DataFrame(), {"enabled": False, "path": PAGASA_DATASET_PATH, "rows": 0}

    try:
        raw = pd.read_csv(PAGASA_DATASET_PATH)
        if raw.empty:
            return pd.DataFrame(), {"enabled": True, "path": PAGASA_DATASET_PATH, "rows": 0}

        normalized = raw.copy()
        normalized.columns = [
            str(col).strip().lower().replace(" ", "_").replace("-", "_")
            for col in normalized.columns
        ]

        alias_map = {
            "timestamp": ["timestamp", "date", "datetime"],
            "total_rain": ["total_rain", "rainfall", "rain_mm", "rain", "precipitation"],
            "water_level": ["water_level", "waterlevel", "wl", "flood_level"],
            "flow_rate": ["flow_rate", "flow", "flowrate", "flow_l"],
            "flooded": ["flooded", "label", "risk", "risk_label", "class"],
            "device_id": ["device_id", "station", "station_id", "source_device"],
        }

        selected = {}
        for target, aliases in alias_map.items():
            match = next((name for name in aliases if name in normalized.columns), None)
            if match:
                selected[target] = normalized[match]

        required = {"timestamp", "total_rain", "water_level", "flooded"}
        if not required.issubset(selected.keys()):
            missing = sorted(required - set(selected.keys()))
            print(f"⚠️ PAGASA dataset missing required columns: {missing}")
            return pd.DataFrame(), {
                "enabled": True,
                "path": PAGASA_DATASET_PATH,
                "rows": 0,
                "error": f"Missing columns: {', '.join(missing)}"
            }

        df = pd.DataFrame(selected)
        if "flow_rate" not in df.columns:
            df["flow_rate"] = 0.0
        if "device_id" not in df.columns:
            df["device_id"] = "pagasa"

        df["source"] = "pagasa"
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["total_rain"] = pd.to_numeric(df["total_rain"], errors="coerce")
        df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
        df["flow_rate"] = pd.to_numeric(df["flow_rate"], errors="coerce").fillna(0.0)
        df["flooded"] = df["flooded"].apply(normalize_flood_label)
        df = df.dropna(subset=["timestamp", "total_rain", "water_level", "flooded"]).copy()
        df["flooded"] = df["flooded"].astype(int)
        df["id"] = -1 * (pd.RangeIndex(start=1, stop=len(df) + 1))
        df = df[["id", "timestamp", "device_id", "total_rain", "water_level", "flow_rate", "flooded", "source"]]

        return df, {
            "enabled": True,
            "path": PAGASA_DATASET_PATH,
            "rows": len(df)
        }
    except Exception as e:
        print(f"⚠️ Could not load PAGASA dataset: {e}")
        return pd.DataFrame(), {
            "enabled": True,
            "path": PAGASA_DATASET_PATH,
            "rows": 0,
            "error": str(e)
        }


def load_labelled_training_data(limit_db=None):
    conn = get_db_connection()
    query = """
        SELECT id, timestamp, device_id, total_rain, water_level, flow_rate, flooded
        FROM sensor_readings
        WHERE flooded IS NOT NULL
        ORDER BY timestamp ASC
    """
    if isinstance(limit_db, int) and limit_db > 0:
        query = f"""
            SELECT id, timestamp, device_id, total_rain, water_level, flow_rate, flooded
            FROM (
                SELECT id, timestamp, device_id, total_rain, water_level, flow_rate, flooded
                FROM sensor_readings
                WHERE flooded IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT {int(limit_db)}
            ) recent_rows
            ORDER BY timestamp ASC
        """

    db_df = pd.read_sql(query, conn)
    conn.close()

    if not db_df.empty:
        db_df["source"] = "database"

    pagasa_df, pagasa_meta = load_pagasa_dataset()

    frames = []
    if not db_df.empty:
        frames.append(db_df)
    if not pagasa_df.empty:
        frames.append(pagasa_df)

    if not frames:
        return pd.DataFrame(), {
            "database_rows": 0,
            "pagasa_rows": pagasa_meta.get("rows", 0),
            "pagasa_enabled": pagasa_meta.get("enabled", False),
            "pagasa_path": pagasa_meta.get("path", PAGASA_DATASET_PATH),
            "pagasa_error": pagasa_meta.get("error"),
        }

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = prepare_feature_frame(combined)
    combined["flooded"] = pd.to_numeric(combined["flooded"], errors="coerce")
    combined = combined[combined["flooded"].isin([0, 1, 2])].copy()

    meta = {
        "database_rows": 0 if db_df.empty else len(db_df),
        "pagasa_rows": pagasa_meta.get("rows", 0),
        "pagasa_enabled": pagasa_meta.get("enabled", False),
        "pagasa_path": pagasa_meta.get("path", PAGASA_DATASET_PATH),
        "pagasa_error": pagasa_meta.get("error"),
    }
    return combined, meta


def ensure_bootstrap_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {BOOTSTRAP_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            reading_id BIGINT NULL,
            device_id VARCHAR(32) NOT NULL,
            timestamp DATETIME NOT NULL,
            total_rain FLOAT NOT NULL DEFAULT 0,
            water_level FLOAT NOT NULL DEFAULT 0,
            flow_rate FLOAT NOT NULL DEFAULT 0,
            rain_3h FLOAT NOT NULL DEFAULT 0,
            rise_rate FLOAT NOT NULL DEFAULT 0,
            flooded TINYINT NOT NULL,
            label_origin VARCHAR(64) NOT NULL DEFAULT 'threshold_bootstrap',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_bootstrap_reading (reading_id, device_id)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()


def ensure_prediction_audit_table():
    global prediction_audit_table_ready
    if prediction_audit_table_ready:
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {PREDICTION_AUDIT_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            device_id VARCHAR(32) NOT NULL,
            prediction_time DATETIME NOT NULL,
            prediction_bucket DATETIME NOT NULL,
            current_risk VARCHAR(16) NOT NULL,
            predicted_risk_1h VARCHAR(16) NOT NULL,
            predicted_conf_1h FLOAT NOT NULL DEFAULT 0,
            predicted_risk_3h VARCHAR(16) NOT NULL,
            predicted_conf_3h FLOAT NOT NULL DEFAULT 0,
            baseline_rainfall FLOAT NOT NULL DEFAULT 0,
            baseline_water_level FLOAT NOT NULL DEFAULT 0,
            baseline_flow_rate FLOAT NOT NULL DEFAULT 0,
            ml_active TINYINT(1) NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_prediction_bucket (device_id, prediction_bucket)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    prediction_audit_table_ready = True


def log_prediction_audit(device_id, sensor, preds, prediction_time, ml_active):
    try:
        ensure_prediction_audit_table()
        conn = get_db_connection()
        cursor = conn.cursor()
        bucket = prediction_bucket_for(prediction_time, 10)
        cursor.execute(
            f"""
            INSERT INTO {PREDICTION_AUDIT_TABLE} (
                device_id, prediction_time, prediction_bucket,
                current_risk, predicted_risk_1h, predicted_conf_1h,
                predicted_risk_3h, predicted_conf_3h,
                baseline_rainfall, baseline_water_level, baseline_flow_rate, ml_active
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                prediction_time = VALUES(prediction_time),
                current_risk = VALUES(current_risk),
                predicted_risk_1h = VALUES(predicted_risk_1h),
                predicted_conf_1h = VALUES(predicted_conf_1h),
                predicted_risk_3h = VALUES(predicted_risk_3h),
                predicted_conf_3h = VALUES(predicted_conf_3h),
                baseline_rainfall = VALUES(baseline_rainfall),
                baseline_water_level = VALUES(baseline_water_level),
                baseline_flow_rate = VALUES(baseline_flow_rate),
                ml_active = VALUES(ml_active)
            """,
            (
                device_id,
                prediction_time,
                bucket,
                preds["current_risk"],
                preds["pred_risk_1h"],
                float(preds["pred_conf_1h"]),
                preds["pred_risk_3h"],
                float(preds["pred_conf_3h"]),
                float(sensor.get("rainfall") or 0),
                float(sensor.get("water_level") or 0),
                float(sensor.get("flow_rate") or 0),
                1 if ml_active else 0,
            )
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Prediction audit log failed: {e}")


def get_actual_row_after(cursor, device_id, target_ts):
    cursor.execute(
        """
        SELECT id, device_id, timestamp, total_rain, water_level, flow_rate, flooded
        FROM sensor_readings
        WHERE device_id = %s AND timestamp >= %s
        ORDER BY timestamp ASC
        LIMIT 1
        """,
        (device_id, target_ts)
    )
    return cursor.fetchone()


def derive_actual_label(cursor, device_id, actual_row):
    stored = normalize_flood_label(actual_row.get("flooded"))
    if stored is not None:
        return stored

    cursor.execute(
        """
        SELECT id, timestamp, device_id, total_rain, water_level, flow_rate, flooded
        FROM sensor_readings
        WHERE device_id = %s AND timestamp <= %s
        ORDER BY timestamp DESC, id DESC
        LIMIT %s
        """,
        (device_id, actual_row["timestamp"], RAIN_3H_WINDOW_READINGS)
    )
    rows = cursor.fetchall()
    if rows:
        rows = list(reversed(rows))
        frame = pd.DataFrame(rows)
        frame = prepare_feature_frame(frame)
        if not frame.empty:
            return int(bootstrap_label_from_features(frame.iloc[-1].to_dict()))

    fallback_risk, _, _ = rule_based_predict({
        "rainfall": float(actual_row.get("total_rain") or 0),
        "water_level": float(actual_row.get("water_level") or 0),
        "flow_rate": float(actual_row.get("flow_rate") or 0),
        "rain_3h": float(actual_row.get("total_rain") or 0),
        "rise_rate": 0.0
    })
    return normalize_flood_label(fallback_risk)


def bootstrap_label_from_features(row):
    wl = float(row.get("water_level", 0) or 0)
    flow = float(row.get("flow_rate", 0) or 0)
    rain_3h = float(row.get("rain_3h", row.get("total_rain", 0)) or 0)
    rise = float(row.get("rise_rate", 0) or 0)

    if wl > 150 or rain_3h > 50 or (rise > 10 and wl > 100):
        return 2
    if wl > 80 or rain_3h > 20 or flow > 3.0:
        return 1
    return 0


def rebuild_bootstrap_dataset():
    ensure_bootstrap_table()

    conn = get_db_connection()
    sensor_df = pd.read_sql(
        """SELECT id, timestamp, device_id, total_rain, water_level, flow_rate
           FROM sensor_readings
           ORDER BY timestamp ASC""",
        conn
    )

    if sensor_df.empty:
        conn.close()
        return {
            "rows_written": 0,
            "class_counts": {0: 0, 1: 0, 2: 0},
            "message": "No sensor_readings rows available to bootstrap."
        }

    sensor_df = prepare_feature_frame(sensor_df)
    sensor_df["flooded"] = sensor_df.apply(bootstrap_label_from_features, axis=1)
    sensor_df["label_origin"] = "threshold_bootstrap"

    rows = []
    for _, row in sensor_df.iterrows():
        ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
        rows.append((
            int(row["id"]) if not pd.isna(row["id"]) else None,
            str(row.get("device_id", "bridge1") or "bridge1"),
            ts,
            float(row.get("total_rain", 0) or 0),
            float(row.get("water_level", 0) or 0),
            float(row.get("flow_rate", 0) or 0),
            float(row.get("rain_3h", 0) or 0),
            float(row.get("rise_rate", 0) or 0),
            int(row.get("flooded", 0) or 0),
            str(row.get("label_origin", "threshold_bootstrap"))
        ))

    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {BOOTSTRAP_TABLE}")
    cursor.executemany(
        f"""
        INSERT INTO {BOOTSTRAP_TABLE}
        (reading_id, device_id, timestamp, total_rain, water_level, flow_rate,
         rain_3h, rise_rate, flooded, label_origin)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        rows
    )
    conn.commit()
    cursor.close()
    conn.close()

    class_counts = sensor_df["flooded"].astype(int).value_counts().sort_index().to_dict()
    return {
        "rows_written": len(rows),
        "class_counts": {0: int(class_counts.get(0, 0)), 1: int(class_counts.get(1, 0)), 2: int(class_counts.get(2, 0))},
        "message": f"Bootstrap dataset rebuilt from {len(rows)} sensor rows."
    }


def load_bootstrap_training_data():
    ensure_bootstrap_table()
    conn = get_db_connection()
    bootstrap_df = pd.read_sql(
        f"""SELECT id, reading_id, timestamp, device_id, total_rain, water_level,
                  flow_rate, rain_3h, rise_rate, flooded, label_origin, updated_at
           FROM {BOOTSTRAP_TABLE}
           ORDER BY timestamp ASC""",
        conn
    )
    conn.close()

    if not bootstrap_df.empty:
        bootstrap_df["source"] = "bootstrap"
        bootstrap_df["timestamp"] = pd.to_datetime(bootstrap_df["timestamp"])
        bootstrap_df["month"] = bootstrap_df["timestamp"].dt.month
        bootstrap_df = bootstrap_df.fillna(0)

    pagasa_df, pagasa_meta = load_pagasa_dataset()
    if not pagasa_df.empty:
        pagasa_df = prepare_feature_frame(pagasa_df)
    frames = []
    if not bootstrap_df.empty:
        frames.append(bootstrap_df)
    if not pagasa_df.empty:
        frames.append(pagasa_df)

    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not combined.empty:
        combined = prepare_feature_frame(combined)
        combined["flooded"] = pd.to_numeric(combined["flooded"], errors="coerce")
        combined = combined[combined["flooded"].isin([0, 1, 2])].copy()

    meta = {
        "bootstrap_rows": 0 if bootstrap_df.empty else len(bootstrap_df),
        "pagasa_rows": pagasa_meta.get("rows", 0),
        "pagasa_enabled": pagasa_meta.get("enabled", False),
        "pagasa_path": pagasa_meta.get("path", PAGASA_DATASET_PATH),
        "pagasa_error": pagasa_meta.get("error"),
    }
    return combined, meta


def evaluate_training_dataset(df):
    if df is None or df.empty:
        return {
            "ready": False,
            "reason": "No training rows available yet."
        }

    work = df.copy()
    work["flooded"] = pd.to_numeric(work["flooded"], errors="coerce")
    work = work[work["flooded"].isin([0, 1, 2])].copy()
    if work.empty:
        return {
            "ready": False,
            "reason": "No valid class labels in training data."
        }

    class_counts = work["flooded"].astype(int).value_counts().sort_index().to_dict()
    if len(class_counts) < 2:
        return {
            "ready": False,
            "reason": "Need at least two flood classes before accuracy can be estimated.",
            "class_counts": class_counts
        }

    min_class_count = min(class_counts.values())
    if min_class_count < 2:
        return {
            "ready": False,
            "reason": "Need at least two rows in each available class before cross-validation can run.",
            "class_counts": class_counts
        }

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    features = ["total_rain", "water_level", "flow_rate", "rain_3h", "rise_rate", "month"]
    X = work[features]
    y = work["flooded"].astype(int)

    n_splits = min(5, int(min_class_count))
    if n_splits < 2:
        return {
            "ready": False,
            "reason": "Not enough class diversity for cross-validation.",
            "class_counts": class_counts
        }

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    return {
        "ready": True,
        "accuracy": round(float(accuracy_score(y, y_pred) * 100), 1),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, y_pred) * 100), 1),
        "evaluation_method": f"{n_splits}-fold stratified cross-validation on training data",
        "class_counts": {int(k): int(v) for k, v in class_counts.items()}
    }


def get_training_mode_status():
    ensure_bootstrap_table()
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN flooded = 0 THEN 1 ELSE 0 END) AS low_rows,
            SUM(CASE WHEN flooded = 1 THEN 1 ELSE 0 END) AS moderate_rows,
            SUM(CASE WHEN flooded = 2 THEN 1 ELSE 0 END) AS high_rows,
            MAX(updated_at) AS last_built_at
        FROM {BOOTSTRAP_TABLE}
    """)
    stats = cursor.fetchone() or {}
    cursor.close()
    conn.close()

    training_df, training_meta = load_bootstrap_training_data()
    model_loaded = load_model() is not None
    evaluation = evaluate_training_dataset(training_df)

    return {
        "model_loaded": model_loaded,
        "bootstrap_rows": int(stats.get("total_rows") or 0),
        "class_counts": {
            0: int(stats.get("low_rows") or 0),
            1: int(stats.get("moderate_rows") or 0),
            2: int(stats.get("high_rows") or 0),
        },
        "last_built_at": stats.get("last_built_at").isoformat() if hasattr(stats.get("last_built_at"), "isoformat") else stats.get("last_built_at"),
        "sources": training_meta,
        "evaluation": evaluation
    }

# =========================================
# FETCH RECENT READINGS PER DEVICE
# =========================================
def get_recent_readings(device_id, n=RAIN_3H_WINDOW_READINGS):
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT total_rain, water_level, flow_rate FROM sensor_readings "
            "WHERE device_id = %s ORDER BY timestamp DESC LIMIT %s",
            (device_id, n)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        rows.reverse()
        return rows
    except Exception as e:
        print(f"⚠️ Could not fetch readings for {device_id}: {e}")
        return []


def using_demo_state():
    return bool(simulation_mode.get("active"))

# =========================================
# GET WEATHER FORECAST FROM OPENWEATHERMAP
# Returns: temp, humidity, next 1h rain estimate,
#          next 3h rain estimate
# =========================================
def get_weather_data(lat, lng):
    result = {
        "temp":           28.0,
        "humidity":       70,
        "forecast_1h":    0.0,
        "forecast_3h":    0.0,
    }
    try:
        # Current weather
        url_now = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lng}&appid={API_KEY}&units=metric"
        )
        r = requests.get(url_now, timeout=5).json()
        result["temp"]     = r["main"]["temp"]
        result["humidity"] = r["main"]["humidity"]
    except Exception:
        pass

    try:
        # Forecast data comes in 3-hour blocks, so 1-hour rain is estimated
        # as one third of the next 3-hour accumulation.
        url_fc = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lng}&appid={API_KEY}&units=metric&cnt=1"
        )
        fc = requests.get(url_fc, timeout=5).json()
        periods = fc.get("list", [])
        if len(periods) >= 1:
            next_3h_rain = float(periods[0].get("rain", {}).get("3h", 0.0) or 0.0)
            result["forecast_3h"] = next_3h_rain
            result["forecast_1h"] = round(next_3h_rain / 3.0, 2)
    except Exception:
        pass

    return result

# =========================================
# RULE-BASED FALLBACK PREDICTION
# =========================================
def rule_based_predict(sensor_data):
    wl   = sensor_data.get("water_level", 0)
    rain = sensor_data.get("rainfall",    0)
    flow = sensor_data.get("flow_rate",   0)
    r3h  = sensor_data.get("rain_3h",     rain)
    rise = sensor_data.get("rise_rate",   0)

    if wl > 150 or r3h > 50 or (rise > 10 and wl > 100):
        return "High",     "red",    85.0
    elif wl > 80 or r3h > 20 or flow > 3.0:
        return "Moderate", "orange", 75.0
    else:
        return "Low",      "green",  90.0

# =========================================
# CORE PREDICTION FUNCTION
# Takes a feature dict, returns risk label
# =========================================
def run_prediction(features, clf):
    """
    features = {
        total_rain, water_level, flow_rate,
        rain_3h, rise_rate, month
    }
    Returns: (risk_label, color, confidence)
    """
    if clf is None:
        return rule_based_predict({
            "rainfall":    features["total_rain"],
            "water_level": features["water_level"],
            "flow_rate":   features["flow_rate"],
            "rain_3h":     features["rain_3h"],
            "rise_rate":   features["rise_rate"]
        })

    df = pd.DataFrame([features])
    try:
        pred          = clf.predict(df)[0]
        probabilities = clf.predict_proba(df)[0]
        confidence    = round(max(probabilities) * 100, 2)
    except Exception as e:
        print(f"⚠️ ML prediction error: {e} — rule-based fallback")
        return rule_based_predict({
            "rainfall":    features["total_rain"],
            "water_level": features["water_level"],
            "flow_rate":   features["flow_rate"],
            "rain_3h":     features["rain_3h"],
            "rise_rate":   features["rise_rate"]
        })

    if pred == 0:   return "Low",      "green",  confidence
    elif pred == 1: return "Moderate", "orange", confidence
    else:           return "High",     "red",    confidence

# =========================================
# PREDICTION ENGINE
# Computes CURRENT risk + PREDICTED risk
# (current + forecast rainfall scenario)
# =========================================
def get_predictions(device_id, forecast_1h, forecast_3h, clf):
    sensor  = latest_sensor_data[device_id].copy()
    history = [] if using_demo_state() else get_recent_readings(device_id, RAIN_3H_WINDOW_READINGS)

    # Compute rolling features from actual history
    if using_demo_state():
        rain_3h = max(float(sensor.get("rain_3h", 0) or 0), float(sensor.get("rainfall", 0) or 0))
        rise_rate = float(sensor.get("rise_rate", 0) or 0)
    elif len(history) >= 2:
        rain_values  = [r["total_rain"]  for r in history]
        water_values = [r["water_level"] for r in history]
        rain_3h   = sum(rain_values[-RAIN_3H_WINDOW_READINGS:]) if len(rain_values) >= RAIN_3H_WINDOW_READINGS else sum(rain_values)
        rise_rate = water_values[-1] - water_values[-2]
    else:
        rain_3h   = sensor["rainfall"]
        rise_rate = 0.0

    # Store computed features for popup display
    latest_sensor_data[device_id]["rain_3h"]   = rain_3h
    latest_sensor_data[device_id]["rise_rate"] = rise_rate

    month = datetime.now().month

    # ── CURRENT RISK ──
    # Based on what sensors are reading right now
    current_features = {
        "total_rain":  sensor["rainfall"],
        "water_level": sensor["water_level"],
        "flow_rate":   sensor["flow_rate"],
        "rain_3h":     rain_3h,
        "rise_rate":   rise_rate,
        "month":       month
    }
    prediction_model = None if using_demo_state() else clf
    current_risk, current_color, current_conf = run_prediction(current_features, prediction_model)

    # ── PREDICTED RISK (1 hour from now) ──
    # Simulates what the sensor will look like in 1h
    # by adding forecast rainfall to current accumulation.
    # This is the actual prediction capability.
    predicted_rain_1h = rain_3h + forecast_1h
    predicted_wl_1h   = sensor["water_level"] + max(0, forecast_1h * 0.4)

    predicted_features_1h = {
        "total_rain":  sensor["rainfall"] + forecast_1h,
        "water_level": predicted_wl_1h,
        "flow_rate":   sensor["flow_rate"],
        "rain_3h":     predicted_rain_1h,
        "rise_rate":   rise_rate + (forecast_1h * 0.1),
        "month":       month
    }
    pred_risk_1h, pred_color_1h, pred_conf_1h = run_prediction(predicted_features_1h, prediction_model)

    # ── PREDICTED RISK (3 hours from now) ──
    predicted_rain_3h = rain_3h + forecast_3h
    predicted_wl_3h   = sensor["water_level"] + max(0, forecast_3h * 0.4)

    predicted_features_3h = {
        "total_rain":  sensor["rainfall"] + forecast_3h,
        "water_level": predicted_wl_3h,
        "flow_rate":   sensor["flow_rate"],
        "rain_3h":     predicted_rain_3h,
        "rise_rate":   rise_rate + (forecast_3h * 0.1),
        "month":       month
    }
    pred_risk_3h, pred_color_3h, pred_conf_3h = run_prediction(predicted_features_3h, prediction_model)

    return {
        "current_risk":  current_risk,
        "current_color": current_color,
        "current_conf":  current_conf,
        "pred_risk_1h":  pred_risk_1h,
        "pred_color_1h": pred_color_1h,
        "pred_conf_1h":  pred_conf_1h,
        "pred_risk_3h":  pred_risk_3h,
        "pred_color_3h": pred_color_3h,
        "pred_conf_3h":  pred_conf_3h,
        "rain_3h":       rain_3h,
        "rise_rate":     rise_rate,
        "forecast_1h":   forecast_1h,
        "forecast_3h":   forecast_3h,
        # Legacy aliases kept for older UI pieces that may still read 3h/6h keys.
        "pred_risk_6h":  pred_risk_3h,
        "pred_color_6h": pred_color_3h,
        "pred_conf_6h":  pred_conf_3h,
        "forecast_6h":   forecast_3h,
    }

# =========================================
# TREND ARROW HELPER
# =========================================
def trend_arrow(current, predicted):
    levels = {"Low": 0, "Moderate": 1, "High": 2}
    c = levels.get(current, 0)
    p = levels.get(predicted, 0)
    if p > c:   return "↑ Rising",   "#ef4444"
    elif p < c: return "↓ Improving","#22c55e"
    else:       return "→ Stable",   "#94a3b8"

# =========================================
# SUBSIDE TIME ESTIMATOR
# Estimates how long flooding will last
# based on risk level, rainfall, water level,
# and drainage capacity of Mandaue City.
#
# Based on:
# - PAGASA flood bulletin guidelines
# - Mandaue City DRRMO average subside times
# - Urban drainage capacity research (PH)
# =========================================
SUBSIDE_MODEL_PATH = "model_subside.pkl"

def load_subside_model():
    if os.path.exists(SUBSIDE_MODEL_PATH):
        try:
            reg = joblib.load(SUBSIDE_MODEL_PATH)
            return reg
        except Exception:
            return None
    return None

def estimate_subside_time(risk_level, sensor_data, preds):
    """
    Returns (display_text, description, color)
    Always returns a human-readable string, never a raw 0.
    Uses ML regression if available, otherwise rule-based.
    Source: PAGASA guidelines + Mandaue City DRRMO records.
    """
    water_level = float(sensor_data.get("water_level") or 0)
    rain_3h     = float(preds.get("rain_3h")     or 0)
    forecast_3h = float(preds.get("forecast_3h") or 0)
    rise_rate   = float(preds.get("rise_rate")   or 0)
    rainfall    = float(sensor_data.get("rainfall") or 0)
    rain_6h     = rain_3h + forecast_3h

    # Try ML regression model first
    reg = load_subside_model()
    if reg is not None:
        try:
            feature_row = {
                "total_rain":  rainfall,
                "water_level": water_level,
                "flow_rate":   float(sensor_data.get("flow_rate") or 0),
                "rain_3h":     rain_3h,
                "rain_6h":     rain_6h,
                "rise_rate":   rise_rate,
                "month":       datetime.now().month
            }
            X_sub = pd.DataFrame([feature_row])
            expected_cols = list(getattr(reg, "feature_names_in_", []))
            if expected_cols:
                for col in expected_cols:
                    if col not in X_sub.columns:
                        X_sub[col] = 0.0
                X_sub = X_sub[expected_cols]
            predicted_hours = float(reg.predict(X_sub)[0])
            if 0.5 <= predicted_hours <= 48:
                h_min = round(predicted_hours, 1)
                h_max = round(predicted_hours + (1 if predicted_hours < 3 else 2), 1)
                if predicted_hours < 2:
                    return f"{h_min}\u2013{h_max} hrs", "Quick drainage expected", "#22c55e"
                elif predicted_hours < 6:
                    return f"{h_min}\u2013{h_max} hrs", "Moderate drainage time", "#f59e0b"
                else:
                    return f"{h_min}\u2013{h_max} hrs", "Extended flooding expected", "#ef4444"
        except Exception as e:
            print(f"⚠️ Subside ML error: {e}")

    # Rule-based — based on PAGASA thresholds and Mandaue DRRMO data
    if risk_level == "Low":
        if rain_3h < 10 and forecast_3h < 5:
            return "None", "No flooding expected under current conditions", "#22c55e"
        elif rain_3h < 20:
            return "< 1 hour", "Minor surface water — drains quickly", "#22c55e"
        else:
            return "1\u20132 hours", "Light flooding — clears within 1\u20132 hours", "#22c55e"

    elif risk_level == "Moderate":
        if forecast_3h > 15:
            return "3\u20136 hours", "More rain incoming — flooding may extend 3\u20136 hours", "#f59e0b"
        elif water_level > 100:
            return "2\u20134 hours", "Elevated water level — expect 2\u20134 hours to drain", "#f59e0b"
        elif rise_rate > 3:
            return "2\u20135 hours", "Water still rising — may take 2\u20135 hours to subside", "#f59e0b"
        else:
            return "1\u20133 hours", "Moderate flooding — typically clears in 1\u20133 hours", "#f59e0b"

    else:
        if forecast_3h > 30:
            return "6\u201312 hours", "Heavy rain forecast — flooding may persist 6\u201312 hours", "#ef4444"
        elif water_level > 150:
            return "4\u20138 hours", "Severe inundation — roads may be blocked 4\u20138 hours", "#ef4444"
        elif rise_rate > 5:
            return "5\u201310 hours", "Rapidly rising water — extended flooding 5\u201310 hours", "#ef4444"
        else:
            return "3\u20137 hours", "Significant flooding — typically subsides in 3\u20137 hours", "#ef4444"


def build_location_payload(loc, clf, timestamp):
    device_id = loc["id"]

    weather      = get_weather_data(loc["lat"], loc["lng"])
    temp         = weather["temp"]
    humidity     = weather["humidity"]
    forecast_1h  = weather["forecast_1h"]
    forecast_3h  = weather["forecast_3h"]
    preds        = get_predictions(device_id, forecast_1h, forecast_3h, clf)
    trend_text, trend_color = trend_arrow(
        preds["current_risk"], preds["pred_risk_1h"]
    )

    sensor = latest_sensor_data[device_id]
    sub_display, sub_desc, sub_color = estimate_subside_time(
        preds["pred_risk_3h"], sensor, preds
    )
    if not using_demo_state():
        log_prediction_audit(device_id, sensor, preds, datetime.now(), clf is not None)
    subside_display = (
        f'<span style="color:{sub_color};font-weight:700">{sub_display}</span>'
    )

    def fc_label(mm):
        if mm >= 10:
            return (
                f'<span style="color:#ef4444;font-weight:700">'
                f'{mm:.2f} mm ⚠️ Heavy</span>'
            )
        if mm >= 2.5:
            return (
                f'<span style="color:#f59e0b;font-weight:700">'
                f'{mm:.2f} mm ⚠️ Moderate</span>'
            )
        return f'<span style="color:#22c55e">{mm:.2f} mm (Light/None)</span>'

    popup = f"""
    <div style="font-size:13px; line-height:1.6;">
        <b style="font-size:15px">{loc['name']}</b><br>
        <span style="font-size:11px;color:#64748b">Sensor: {device_id}</span><br><br>

        <div style="background:#0f172a;border-radius:8px;padding:10px;margin-bottom:8px">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:4px">📍 Current Risk</div>
            <b style="font-size:18px;color:{preds['current_color']}">
                {preds['current_risk']} Risk
            </b>
            <span style="font-size:11px;color:#64748b;margin-left:6px">
                {preds['current_conf']}% confidence
            </span>
        </div>

        <div style="background:#0f172a;border-radius:8px;padding:10px;margin-bottom:8px;
                    border-left:3px solid {preds['pred_color_1h']}">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:4px">🔮 Predicted — Next 1 Hour</div>
            <b style="font-size:16px;color:{preds['pred_color_1h']}">
                {preds['pred_risk_1h']} Risk
            </b>
            <span style="font-size:11px;color:#64748b;margin-left:6px">
                {preds['pred_conf_1h']}% confidence
            </span><br>
            <span style="font-size:11px;color:{trend_color}">
                {trend_text}
            </span>
        </div>

        <div style="background:#0f172a;border-radius:8px;padding:10px;margin-bottom:8px;
                    border-left:3px solid {preds['pred_color_3h']}">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:4px">🔮 Predicted — Next 3 Hours</div>
            <b style="font-size:14px;color:{preds['pred_color_3h']}">
                {preds['pred_risk_3h']} Risk
            </b>
            <span style="font-size:11px;color:#64748b;margin-left:6px">
                {preds['pred_conf_3h']}% confidence
            </span>
        </div>

        <div style="background:#0f172a;border-radius:8px;padding:10px;margin-bottom:8px;
                    border-left:3px solid {sub_color}">
            <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                        letter-spacing:0.06em;margin-bottom:4px">⏱ Estimated Flood Subside Time</div>
            <b style="font-size:15px">{subside_display}</b><br>
            <span style="font-size:11px;color:#94a3b8">{sub_desc}</span>
        </div>

        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px">
            <b style="color:#e2e8f0">── Sensor Readings ──</b><br>
            🌧 Rainfall (now): {sensor['rainfall']:.2f} mm<br>
            🌧 Rainfall (3h):  {preds['rain_3h']:.2f} mm<br>
            📏 Water Level:    {sensor['water_level']:.2f} cm<br>
            📈 Rise Rate:      {preds['rise_rate']:.2f} cm/reading<br>
            💧 Flow Rate:      {sensor['flow_rate']:.2f} L
        </div>

        <div style="font-size:11px;color:#94a3b8;margin-bottom:4px">
            <b style="color:#e2e8f0">── Weather Forecast ──</b><br>
            🔮 Rain (next 1h): {fc_label(forecast_1h)}<br>
            🔮 Rain (next 3h): {fc_label(forecast_3h)}<br>
            🌡 Temperature:    {temp:.1f} °C<br>
            💧 Humidity:       {humidity}%
        </div>

        <div style="font-size:10px;color:#475569;margin-top:6px">
            🕒 Updated: {timestamp}<br>
            🌲 Prediction: Random Forest + OpenWeatherMap Forecast
        </div>
    </div>
    """

    marker_color = preds["current_color"] if using_demo_state() else preds["pred_color_1h"]

    return {
        "id":               device_id,
        "name":             loc["name"],
        "lat":              loc["lat"],
        "lng":              loc["lng"],
        "risk_color":       marker_color,
        "popup_html":       popup,
        "current_risk":     preds["current_risk"],
        "current_color":    preds["current_color"],
        "current_conf":     preds["current_conf"],
        "pred_risk_1h":     preds["pred_risk_1h"],
        "pred_color_1h":    preds["pred_color_1h"],
        "pred_conf_1h":     preds["pred_conf_1h"],
        "pred_risk_3h":     preds["pred_risk_3h"],
        "pred_color_3h":    preds["pred_color_3h"],
        "pred_conf_3h":     preds["pred_conf_3h"],
        "pred_risk_6h":     preds["pred_risk_6h"],
        "pred_color_6h":    preds["pred_color_6h"],
        "pred_conf_6h":     preds["pred_conf_6h"],
        "confidence":       preds["current_conf"],
        "water_level":      sensor["water_level"],
        "rainfall":         sensor["rainfall"],
        "flow_rate":        sensor["flow_rate"],
        "rise_rate":        preds["rise_rate"],
        "rain_3h":          preds["rain_3h"],
        "forecast_1h":      forecast_1h,
        "forecast_3h":      forecast_3h,
        "forecast_6h":      forecast_3h,
        "temperature":      temp,
        "humidity":         humidity,
        "subside_display":  sub_display,
        "subside_desc":     sub_desc,
        "subside_color":    sub_color,
        "trend_text":       trend_text,
        "trend_color":      trend_color,
        "updated_at":       timestamp,
    }


def get_live_location_payloads():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clf = load_model()
    payloads = [build_location_payload(loc, clf, timestamp) for loc in MONITORED_SITES]
    return payloads, clf is not None


def apply_sensor_row(device_id, row):
    latest_sensor_data[device_id]["water_level"] = float(row.get("water_level") or 0)
    latest_sensor_data[device_id]["rainfall"]    = float(row.get("total_rain") or 0)
    latest_sensor_data[device_id]["flow_rate"]   = float(row.get("flow_rate") or 0)
    flooded = int(row.get("flooded") or 0) if row.get("flooded") is not None else 0
    latest_sensor_data[device_id]["rain_3h"]     = float(row.get("rain_3h", row.get("total_rain", 0)) or 0)
    latest_sensor_data[device_id]["rise_rate"]   = 6.0 if flooded >= 2 else (3.0 if flooded == 1 else 0.0)


def reset_sensor_row(device_id):
    latest_sensor_data[device_id]["water_level"] = 0.0
    latest_sensor_data[device_id]["rainfall"] = 0.0
    latest_sensor_data[device_id]["flow_rate"] = 0.0
    latest_sensor_data[device_id]["rain_3h"] = 0.0
    latest_sensor_data[device_id]["rise_rate"] = 0.0


def apply_simulation_values(device_id, values, level):
    latest_sensor_data[device_id]["water_level"] = float(values.get("water_level") or 0)
    latest_sensor_data[device_id]["rainfall"] = float(values.get("rainfall") or 0)
    latest_sensor_data[device_id]["flow_rate"] = float(values.get("flow_rate") or 0)
    latest_sensor_data[device_id]["rain_3h"] = float(values.get("rain_3h", values.get("rainfall", 0)) or 0)
    latest_sensor_data[device_id]["rise_rate"] = (
        2.0 if level == "moderate" else (5.0 if level == "high" else 0.0)
    )


def insert_sensor_reading(device_id, rainfall, water_level, flow_rate):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sensor_readings (device_id, total_rain, water_level, flow_rate) "
            "VALUES (%s, %s, %s, %s)",
            (device_id, rainfall, water_level, flow_rate)
        )
        conn.commit()
        print(f"✅ DB Insert [{device_id}]: rain={rainfall}, water={water_level}, flow={flow_rate}")
        return {"status": "saved"}
    except mysql.connector.Error as e:
        print(f"❌ MySQL Error: {e}")
        return {"status": "error", "error": str(e)}
    except Exception as e:
        print(f"❌ Unknown DB Error: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def resolve_simulation_targets(target):
    target = (target or "both").lower()
    if target in SIMULATION_BRIDGE_IDS:
        return [target]
    return list(SIMULATION_BRIDGE_IDS)


def restore_live_rows(device_ids):
    restored = []
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        for device_id in device_ids:
            cursor.execute(
                """
                SELECT id, device_id, timestamp, total_rain, water_level, flow_rate, flooded
                FROM sensor_readings
                WHERE device_id = %s
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (device_id,)
            )
            row = cursor.fetchone()
            if row:
                apply_sensor_row(device_id, row)
                restored.append({
                    "device_id": device_id,
                    "source": "database",
                    "reading_id": row.get("id"),
                    "timestamp": row["timestamp"].isoformat() if hasattr(row.get("timestamp"), "isoformat") else row.get("timestamp")
                })
            else:
                reset_sensor_row(device_id)
                restored.append({
                    "device_id": device_id,
                    "source": "empty"
                })
    except Exception as e:
        print(f"⚠️ Could not restore live rows: {e}")
        for device_id in device_ids:
            reset_sensor_row(device_id)
            restored.append({
                "device_id": device_id,
                "source": "empty"
            })
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return restored


def get_latest_row_before(cursor, device_id, target_ts):
    cursor.execute(
        """
        SELECT id, device_id, timestamp, total_rain, water_level, flow_rate, flooded
        FROM sensor_readings
        WHERE device_id = %s AND timestamp <= %s
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (device_id, target_ts)
    )
    row = cursor.fetchone()
    if row:
        return row

    cursor.execute(
        """
        SELECT id, device_id, timestamp, total_rain, water_level, flow_rate, flooded
        FROM sensor_readings
        WHERE device_id = %s
        ORDER BY timestamp ASC
        LIMIT 1
        """,
        (device_id,)
    )
    return cursor.fetchone()


# =========================================
# AUTO-RETRAIN ENDPOINT
# =========================================
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    try:
        print("🔄 Auto-retrain triggered...")
        df, source_meta = load_bootstrap_training_data()

        if len(df) < 30:
            return jsonify({"status": "skipped",
                            "reason": f"Only {len(df)} training rows — need at least 30",
                            "sources": source_meta}), 200

        if df.empty or df["flooded"].nunique() < 2:
            return jsonify({"status": "skipped",
                            "reason": "Need training rows from at least 2 flood classes",
                            "sources": source_meta}), 200

        class_counts = df["flooded"].astype(int).value_counts().sort_index().to_dict()
        if min(class_counts.values()) < 2:
            return jsonify({"status": "skipped",
                            "reason": f"Need at least 2 rows per class for retraining. Class counts: {class_counts}",
                            "sources": source_meta}), 200

        from sklearn.ensemble import RandomForestClassifier
        features = ["total_rain", "water_level", "flow_rate",
                    "rain_3h", "rise_rate", "month"]
        X = df[features]
        y = df["flooded"].astype(int)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42
        )
        clf.fit(X, y)
        joblib.dump(clf, MODEL_PATH)

        msg = f"✅ Retrained on {len(df)} training rows — class counts: {class_counts}"
        print(msg)
        return jsonify({
            "status": "success",
            "message": msg,
            "rows": len(df),
            "class_counts": class_counts,
            "sources": source_meta
        }), 200

    except Exception as e:
        print(f"❌ Retrain error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bootstrap-dataset", methods=["POST"])
def bootstrap_dataset():
    try:
        result = rebuild_bootstrap_dataset()
        status = get_training_mode_status()
        return jsonify({
            "status": "success",
            "message": result["message"],
            "rows_written": result["rows_written"],
            "class_counts": result["class_counts"],
            "training_status": status
        }), 200
    except Exception as e:
        print(f"❌ Bootstrap dataset error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/training-mode-status")
def training_mode_status():
    try:
        return jsonify(get_training_mode_status())
    except Exception as e:
        print(f"❌ Training status error: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================
# ESP32 SENSOR ENDPOINT
# =========================================
@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    device_id   = data.get("deviceId", "bridge1")
    water_level = float(data.get("waterLevel", 0))
    rainfall    = float(data.get("totalRain",  0))
    flow_rate   = float(data.get("flowRate",   0))

    if device_id not in latest_sensor_data:
        device_id = "bridge1"

    latest_sensor_data[device_id]["water_level"] = water_level
    latest_sensor_data[device_id]["rainfall"]    = rainfall
    latest_sensor_data[device_id]["flow_rate"]   = flow_rate

    insert_sensor_reading(device_id, rainfall, water_level, flow_rate)

    return jsonify({"status": "received", "device": device_id})

# =========================================
# MAP DATA ENDPOINT
# =========================================
@app.route("/api/map-data")
def map_data():
    locations, ml_active = get_live_location_payloads()
    return jsonify({"locations": locations, "ml_active": ml_active})

# =========================================
# DASHBOARD PAGE + API
# =========================================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

def compute_actual_vs_predicted_payload():
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    conn = get_db_connection()
    sensor_df = pd.read_sql(
        """
        SELECT id, timestamp, device_id, total_rain, water_level, flow_rate, flooded
        FROM sensor_readings
        WHERE device_id IN ('bridge1', 'bridge2')
        ORDER BY device_id, timestamp ASC, id ASC
        """,
        conn
    )
    conn.close()

    if sensor_df.empty:
        return {
            "setup_required": True,
            "message": "No bridge readings are available yet for historical prediction backtesting.",
            "evaluation_method": "Historical backtest against later actual bridge readings",
            "rows": [],
        }

    sensor_df = prepare_feature_frame(sensor_df)
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    sensor_df["interval_minutes"] = (
        sensor_df.groupby("device_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )
    sensor_df["interval_minutes"] = sensor_df.groupby("device_id")["interval_minutes"].transform(
        lambda s: s.fillna(s.dropna().median() if not s.dropna().empty else 10)
    ).fillna(10).clip(lower=1)

    clf = load_model()
    scored_rows = []
    actual_numeric = []
    predicted_numeric = []
    horizon_stats = {
        "1h": {"correct": 0, "total": 0, "pending": 0},
        "3h": {"correct": 0, "total": 0, "pending": 0},
    }

    for device_id, group in sensor_df.groupby("device_id", sort=False):
        group = group.reset_index(drop=True)
        times = group["timestamp"].to_numpy()
        for _, row in group.iterrows():
            base_time = row["timestamp"]
            interval_minutes = float(row.get("interval_minutes") or 10)
            rainfall_rate_per_hour = max(0.0, float(row.get("total_rain") or 0) * (60.0 / interval_minutes))

            month = int(pd.Timestamp(base_time).month)
            for horizon_key, horizon_hours in (("1h", 1), ("3h", 3)):
                target_time = pd.Timestamp(base_time) + pd.Timedelta(hours=horizon_hours)
                target_index = times.searchsorted(target_time.to_datetime64(), side="left")
                if target_index >= len(group):
                    horizon_stats[horizon_key]["pending"] += 1
                    continue

                future_row = group.iloc[int(target_index)]
                actual_value = normalize_flood_label(future_row.get("flooded"))
                if actual_value is None:
                    actual_value = int(bootstrap_label_from_features(future_row.to_dict()))

                predicted_additional_rain = rainfall_rate_per_hour * horizon_hours
                predicted_features = {
                    "total_rain": float(row.get("total_rain") or 0) + predicted_additional_rain,
                    "water_level": float(row.get("water_level") or 0) + max(0.0, predicted_additional_rain * 0.4),
                    "flow_rate": float(row.get("flow_rate") or 0),
                    "rain_3h": float(row.get("rain_3h") or 0) + predicted_additional_rain,
                    "rise_rate": float(row.get("rise_rate") or 0) + (predicted_additional_rain * 0.1),
                    "month": month,
                }
                predicted_label, _, confidence = run_prediction(predicted_features, clf)
                predicted_value = normalize_flood_label(predicted_label)
                if predicted_value is None:
                    predicted_value = 0

                is_match = predicted_value == actual_value
                horizon_stats[horizon_key]["total"] += 1
                if is_match:
                    horizon_stats[horizon_key]["correct"] += 1

                actual_numeric.append(actual_value)
                predicted_numeric.append(predicted_value)
                scored_rows.append({
                    "prediction_time": base_time.isoformat(),
                    "actual_time": future_row["timestamp"].isoformat() if hasattr(future_row["timestamp"], "isoformat") else str(future_row["timestamp"]),
                    "device_id": device_id,
                    "horizon": horizon_key,
                    "actual": flood_label_text(actual_value),
                    "predicted": flood_label_text(predicted_value),
                    "confidence": round(float(confidence), 1),
                    "match": is_match,
                })

    if not scored_rows:
        return {
            "setup_required": True,
            "message": "Historical predictions are not ready yet because the dataset does not have enough later readings for 1-hour or 3-hour comparisons.",
            "evaluation_method": "Historical backtest against later actual bridge readings",
            "rows": [],
        }

    class_counts = pd.Series(actual_numeric).value_counts().sort_index().to_dict()
    prediction_counts = pd.Series(predicted_numeric).value_counts().sort_index().to_dict()
    correct_by_class = {
        "low_correct": int(sum(1 for a, p in zip(actual_numeric, predicted_numeric) if a == 0 and p == 0)),
        "mod_correct": int(sum(1 for a, p in zip(actual_numeric, predicted_numeric) if a == 1 and p == 1)),
        "high_correct": int(sum(1 for a, p in zip(actual_numeric, predicted_numeric) if a == 2 and p == 2)),
    }
    wrong_by_class = {
        "low_wrong": int(sum(1 for a, p in zip(actual_numeric, predicted_numeric) if a == 0 and p != 0)),
        "mod_wrong": int(sum(1 for a, p in zip(actual_numeric, predicted_numeric) if a == 1 and p != 1)),
        "high_wrong": int(sum(1 for a, p in zip(actual_numeric, predicted_numeric) if a == 2 and p != 2)),
    }

    total = len(actual_numeric)
    correct = int(sum(1 for a, p in zip(actual_numeric, predicted_numeric) if a == p))
    accuracy = round(float(accuracy_score(actual_numeric, predicted_numeric) * 100), 1)
    balanced = None
    if len(set(actual_numeric)) >= 2:
        balanced = round(float(balanced_accuracy_score(actual_numeric, predicted_numeric) * 100), 1)

    horizons = {}
    for key, values in horizon_stats.items():
        h_total = values["total"]
        h_correct = values["correct"]
        horizons[key] = {
            "accuracy": round((h_correct / h_total) * 100, 1) if h_total else None,
            "correct": h_correct,
            "total": h_total,
            "pending": values["pending"],
        }

    scored_rows.sort(key=lambda item: item["prediction_time"], reverse=True)

    return {
        "setup_required": False,
        "message": "Historical backtest compares each 1-hour and 3-hour prediction against the later actual bridge reading gathered after that prediction window.",
        "evaluation_method": "Historical backtest on all past bridge readings using later actual gathered data",
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "correct": correct,
        "total": total,
        "class_counts": class_counts,
        "prediction_counts": prediction_counts,
        "correct_by_class": correct_by_class,
        "wrong_by_class": wrong_by_class,
        "horizons": horizons,
        "rows": scored_rows[:30],
    }

@app.route("/api/dashboard-data")
def dashboard_data():
    try:
        live_locations, ml_active = get_live_location_payloads()
        live_by_bridge = {loc["id"]: loc for loc in live_locations}
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT * FROM sensor_readings WHERE device_id = 'bridge1' "
            "ORDER BY timestamp DESC LIMIT 50"
        )
        bridge1 = list(reversed(cursor.fetchall()))

        cursor.execute(
            "SELECT * FROM sensor_readings WHERE device_id = 'bridge2' "
            "ORDER BY timestamp DESC LIMIT 50"
        )
        bridge2 = list(reversed(cursor.fetchall()))

        cursor.execute(
            "SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 20"
        )
        recent = cursor.fetchall()

        cursor.execute("""
            SELECT
                COUNT(*)                                       AS total,
                SUM(CASE WHEN flooded=2 THEN 1 ELSE 0 END)   AS high,
                SUM(CASE WHEN flooded=1 THEN 1 ELSE 0 END)   AS moderate,
                AVG(CASE WHEN device_id='bridge1'
                         THEN water_level END)                AS avg_water_level,
                MAX(total_rain)                               AS max_rain
            FROM sensor_readings
        """)
        stats = cursor.fetchone()
        cursor.close()
        conn.close()

        def serialize(rows):
            out = []
            for r in rows:
                row = dict(r)
                if hasattr(row.get('timestamp'), 'isoformat'):
                    row['timestamp'] = row['timestamp'].isoformat()
                out.append(row)
            return out

        payload = {
            "bridge1": serialize(bridge1),
            "bridge2": serialize(bridge2),
            "bridges": live_by_bridge,
            "recent":  serialize(recent),
            "stats":   {k: (float(v) if v is not None else 0)
                        for k, v in stats.items()},
            "model": {
                "ml_active": ml_active,
                "mode_label": "Random Forest" if ml_active else "Threshold Rules",
                "maps_key_loaded": bool(GOOGLE_MAPS_API_KEY)
            }
        }
        if request.args.get("include_comparison") == "1":
            payload["actual_vs_predicted"] = compute_actual_vs_predicted_payload()
        return jsonify(payload)

    except Exception as e:
        print(f"❌ Dashboard data error: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================
# MAIN PAGE
# =========================================

# =========================================
# SIMULATE SENSOR DATA ENDPOINT
# Injects fake sensor readings so the map
# can demo Low / Moderate / High risk
# =========================================
SIMULATION_PRESETS = {
    "low": {
        "bridge1": {"water_level": 25.0,  "rainfall": 0.5,  "flow_rate": 0.1},
        "bridge2": {"water_level": 22.0,  "rainfall": 0.3,  "flow_rate": 0.05},
    },
    "moderate": {
        "bridge1": {"water_level": 95.0,  "rainfall": 18.0, "flow_rate": 2.5},
        "bridge2": {"water_level": 88.0,  "rainfall": 16.0, "flow_rate": 2.1},
    },
    "high": {
        "bridge1": {"water_level": 165.0, "rainfall": 38.0, "flow_rate": 5.2},
        "bridge2": {"water_level": 158.0, "rainfall": 35.0, "flow_rate": 4.8},
    },
    "reset": {
        "bridge1": {"water_level": 0.0,   "rainfall": 0.0,  "flow_rate": 0.0},
        "bridge2": {"water_level": 0.0,   "rainfall": 0.0,  "flow_rate": 0.0},
    }
}

simulation_mode = {
    "active": False,
    "level": None,
    "target": "both",
    "saved": False,
    "saved_device_ids": []
}

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json(silent=True) or {}
    level = str(data.get("level", "low")).lower()
    target = str(data.get("target", "both")).lower()
    save_to_db = bool(data.get("saveToDb", False))
    custom_device_id = str(data.get("deviceId", "") or "").strip()

    if level not in SIMULATION_PRESETS:
        return jsonify({"error": f"Invalid level. Choose: {list(SIMULATION_PRESETS.keys())}"}), 400
    if target not in (*SIMULATION_BRIDGE_IDS, "both"):
        return jsonify({"error": "Invalid target. Choose bridge1, bridge2, or both."}), 400

    target_ids = resolve_simulation_targets(target)

    if level == "reset":
        restored = restore_live_rows(target_ids)
        simulation_mode["active"] = False
        simulation_mode["level"] = None
        simulation_mode["target"] = target
        simulation_mode["saved"] = False
        simulation_mode["saved_device_ids"] = []

        target_label = "both bridges" if target == "both" else ("Bridge 1" if target == "bridge1" else "Bridge 2")
        return jsonify({
            "status": "ok",
            "level": level,
            "target": target,
            "message": f"Reset {target_label} to the latest live database readings",
            "restored": restored,
            "save_to_db": False,
            "saved_rows": []
        })

    preset = SIMULATION_PRESETS[level]
    applied = []
    saved_rows = []

    for device_id in target_ids:
        values = preset.get(device_id, {})
        apply_simulation_values(device_id, values, level)
        applied.append({
            "bridge_id": device_id,
            "water_level": float(values.get("water_level") or 0),
            "rainfall": float(values.get("rainfall") or 0),
            "flow_rate": float(values.get("flow_rate") or 0)
        })

        if save_to_db:
            if custom_device_id:
                db_device_id = custom_device_id if len(target_ids) == 1 else f"{custom_device_id}_{device_id}"
            else:
                db_device_id = f"sim_{level}_{device_id}"
            save_result = insert_sensor_reading(
                db_device_id,
                float(values.get("rainfall") or 0),
                float(values.get("water_level") or 0),
                float(values.get("flow_rate") or 0)
            )
            saved_rows.append({
                "device_id": db_device_id,
                "bridge_id": device_id,
                **save_result
            })

    simulation_mode["active"] = True
    simulation_mode["level"] = level
    simulation_mode["target"] = target
    simulation_mode["saved"] = save_to_db
    simulation_mode["saved_device_ids"] = [row["device_id"] for row in saved_rows if row.get("status") == "saved"]

    target_label = {
        "bridge1": "Bridge 1",
        "bridge2": "Bridge 2",
        "both": "both bridges"
    }.get(target, "both bridges")
    save_note = ""
    if save_to_db:
        saved_ids = simulation_mode["saved_device_ids"]
        save_note = f" and saved to DB as {', '.join(saved_ids)}" if saved_ids else " and attempted to save to DB"

    print(f"🎭 Simulation mode: level={level}, target={target}, save_to_db={save_to_db}")
    return jsonify({
        "status": "ok",
        "level": level,
        "target": target,
        "applied": applied,
        "save_to_db": save_to_db,
        "saved_rows": saved_rows,
        "message": f"Simulating {level.upper()} flood risk on {target_label}{save_note}"
    })

@app.route("/api/simulation-status")
def simulation_status():
    return jsonify(simulation_mode)

# =========================================
# HISTORICAL EVENTS ENDPOINT
# Returns past Moderate/High readings
# from the database for replay
# =========================================
@app.route("/api/historical-events")
def historical_events():
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get replayable Moderate/High events, prioritizing actual bridge rows.
        cursor.execute("""
            SELECT
                id, device_id, timestamp,
                total_rain, water_level, flow_rate, flooded
            FROM sensor_readings
            WHERE flooded >= 1
            ORDER BY
                CASE WHEN device_id IN ('bridge1', 'bridge2') THEN 0 ELSE 1 END,
                timestamp DESC
            LIMIT 50
        """)
        events = cursor.fetchall()
        cursor.close()
        conn.close()

        serialized = []
        for e in events:
            row = dict(e)
            if hasattr(row.get("timestamp"), "isoformat"):
                row["timestamp"] = row["timestamp"].isoformat()
            row["risk_label"] = ["Low", "Moderate", "High"][min(int(row["flooded"]), 2)]
            serialized.append(row)

        return jsonify({"events": serialized, "count": len(serialized)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
# REPLAY HISTORICAL EVENT ENDPOINT
# Injects a past event's sensor values
# into the live system for demo/testing
# =========================================
@app.route("/api/replay-event", methods=["POST"])
def replay_event():
    data     = request.get_json()
    event_id = data.get("id")

    if not event_id:
        return jsonify({"error": "No event id provided"}), 400

    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM sensor_readings WHERE id = %s", (event_id,)
        )
        event = cursor.fetchone()

        if not event:
            cursor.close()
            conn.close()
            return jsonify({"error": "Event not found"}), 404

        # Replay the selected event into both bridge markers so the map visibly
        # reflects the chosen Moderate / High history item.
        replay_rows = []
        for bridge_id in latest_sensor_data:
            replay_row = dict(event)
            replay_row["device_id"] = bridge_id
            apply_sensor_row(bridge_id, replay_row)
            replay_rows.append({
                "device_id": bridge_id,
                "reading_id": replay_row["id"],
                "timestamp": replay_row["timestamp"].isoformat() if hasattr(replay_row["timestamp"], "isoformat") else replay_row["timestamp"],
                "water_level": float(replay_row["water_level"] or 0),
                "total_rain": float(replay_row["total_rain"] or 0),
                "flow_rate": float(replay_row["flow_rate"] or 0),
                "flooded": int(replay_row["flooded"] or 0) if replay_row.get("flooded") is not None else 0
            })

        simulation_mode["active"] = True
        simulation_mode["level"] = f"replay:{event_id}"
        simulation_mode["target"] = "both"
        simulation_mode["saved"] = False
        simulation_mode["saved_device_ids"] = []

        ts = event["timestamp"]
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()

        cursor.close()
        conn.close()

        return jsonify({
            "status":      "ok",
            "device_id":   event.get("device_id", "bridge1"),
            "timestamp":   ts,
            "water_level": float(event["water_level"] or 0),
            "total_rain":  float(event["total_rain"]  or 0),
            "flow_rate":   float(event["flow_rate"]   or 0),
            "flooded":     int(event["flooded"]       or 0),
            "replayed":    replay_rows,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
# ACTUAL VS PREDICTED ENDPOINT
# Runs the ML model on historical DB rows
# and compares to their stored labels
# =========================================
@app.route("/api/actual-vs-predicted")
def actual_vs_predicted():
    try:
        return jsonify(compute_actual_vs_predicted_payload()), 200
    except Exception as e:
        print(f"❌ Actual vs Predicted error: {e}")
        return jsonify({
            "setup_required": True,
            "message": "Actual vs Predicted is temporarily unavailable.",
            "accuracy": 0.0,
            "balanced_accuracy": None,
            "correct": 0,
            "total": 0,
            "confusion": {
                "low_correct": 0,
                "mod_correct": 0,
                "high_correct": 0,
                "low_as_moderate": 0,
                "low_as_high": 0,
                "mod_as_low": 0,
                "mod_as_high": 0,
                "high_as_low": 0,
                "high_as_moderate": 0,
            },
            "rows": [],
            "class_counts": {},
            "prediction_counts": {},
            "evaluation_method": str(e),
            "sources": {},
            "error": str(e),
        }), 200

@app.route("/")
def home():
    return render_template("map.html", google_maps_api_key=GOOGLE_MAPS_API_KEY)

# =========================================
# RUN APP
# =========================================
if __name__ == "__main__":
    print("FloodGuide Running...")
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        t = threading.Thread(target=keep_alive, daemon=True)
        t.start()
    app.run(host="0.0.0.0", port=5000, debug=True)
