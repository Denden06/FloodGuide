import pandas as pd
import numpy as np
import joblib
import mysql.connector
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

# ===============================
# DATABASE CONNECTION
# ===============================
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQLHOST", "caboose.proxy.rlwy.net"),
        user=os.getenv("MYSQLUSER", "root"),
        password=os.getenv("MYSQLPASSWORD", "cYHockIgVucbKAkgNvkRjTqsTGngkjvD"),
        database=os.getenv("MYSQLDATABASE", "railway"),
        port=int(os.getenv("MYSQLPORT", 52603))
    )

# ===============================
# LOAD DATA FROM MYSQL
# ===============================
def load_sensor_data():
    try:
        conn = get_db_connection()
        query = "SELECT * FROM sensor_readings ORDER BY timestamp ASC"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        return pd.DataFrame()

# ===============================
# FEATURE ENGINEERING
# ===============================
def prepare_features(df):
    if df.empty:
        print("No data to train on!")
        return None, None, None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Rolling rainfall
    df['rain_3h'] = df['total_rain'].rolling(window=3).sum()
    df['rain_6h'] = df['total_rain'].rolling(window=6).sum()

    # Water level rise rate
    df['rise_rate'] = df['water_level'].diff().fillna(0)

    # Month
    df['month'] = df['timestamp'].dt.month

    # Fill missing values
    df = df.fillna(0)

    features = ['total_rain','water_level','flow_rate','rain_3h','rain_6h','rise_rate','month']

    X = df[features]

    # Assuming you have these columns in the table (or set defaults)
    if 'flooded' not in df.columns:
        df['flooded'] = 0
    if 'subside_time' not in df.columns:
        df['subside_time'] = 0

    y_class = df['flooded']
    y_reg = df['subside_time']

    return X, y_class, y_reg

# ===============================
# TRAIN MODELS
# ===============================
def train_models(X, y_class, y_reg):
    X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    clf.fit(X_train, y_class_train)
    y_class_pred = clf.predict(X_test)
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_class_test, y_class_pred))

    reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    print("\n===== REGRESSION METRICS =====")
    print("MAE:", mean_absolute_error(y_reg_test, y_reg_pred))
    print("R2:", r2_score(y_reg_test, y_reg_pred))

    # Save models
    joblib.dump(clf, "model_class.pkl")
    joblib.dump(reg, "model_subside.pkl")
    print("\n✅ Models retrained and saved successfully!")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("Loading sensor data from MySQL...")
    df = load_sensor_data()

    X, y_class, y_reg = prepare_features(df)

    if X is not None:
        print("Training models...")
        train_models(X, y_class, y_reg)
    else:
        print("❌ No data available for training.")