import pandas as pd
import numpy as np
import joblib
import mysql.connector
import os
import sys
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
        port=int(os.getenv("MYSQLPORT", 52603)),
        connection_timeout=10
    )

# ===============================
# STEP 1 — MIGRATE DB TABLE
# Adds flooded + subside_time columns
# if they don't exist yet. Safe to run
# multiple times.
# ===============================
def migrate_table():
    print("🔧 Checking database schema...")
    conn   = get_db_connection()
    cursor = conn.cursor()

    migrations = [
        # Add flooded column (0=Low, 1=Moderate, 2=High)
        """
        ALTER TABLE sensor_readings
        ADD COLUMN IF NOT EXISTS flooded TINYINT NOT NULL DEFAULT 0
        """,
        # Add subside_time column (hours until water recedes)
        """
        ALTER TABLE sensor_readings
        ADD COLUMN IF NOT EXISTS subside_time FLOAT NOT NULL DEFAULT 0
        """,
    ]

    for sql in migrations:
        try:
            cursor.execute(sql)
            conn.commit()
        except mysql.connector.Error as e:
            # Column may already exist on older MySQL — that's fine
            if e.errno != 1060:  # 1060 = Duplicate column name
                print(f"  ⚠️ Migration warning: {e}")

    cursor.close()
    conn.close()
    print("✅ Schema is ready.\n")

# ===============================
# STEP 2 — AUTO-LABEL DATA
# If flooded column is all zeros,
# apply rule-based labelling from
# Mandaue flood thresholds so the
# model has something meaningful
# to learn before PAGASA data
# arrives.
# ===============================
def auto_label_if_needed():
    print("🏷️  Checking if flood labels exist...")
    conn   = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sensor_readings WHERE flooded > 0")
    labelled_count = cursor.fetchone()[0]

    if labelled_count > 0:
        print(f"  ✅ {labelled_count} labelled rows found — skipping auto-label.\n")
        cursor.close()
        conn.close()
        return

    print("  ⚠️  No labels found — applying rule-based auto-labelling...")
    print("  (You should replace these with real PAGASA-verified labels later)\n")

    # Fetch all rows
    cursor.execute("SELECT id, total_rain, water_level, flow_rate FROM sensor_readings")
    rows = cursor.fetchall()

    updates = []
    for (row_id, rain, wl, flow) in rows:
        rain = rain or 0
        wl   = wl   or 0
        flow = flow or 0

        # Rule-based thresholds for Mandaue City
        if wl > 150 or rain > 30:
            label = 2   # High
        elif wl > 80 or rain > 15 or flow > 3.0:
            label = 1   # Moderate
        else:
            label = 0   # Low

        updates.append((label, row_id))

    if updates:
        cursor.executemany(
            "UPDATE sensor_readings SET flooded = %s WHERE id = %s", updates
        )
        conn.commit()
        counts = {0: 0, 1: 0, 2: 0}
        for (label, _) in updates:
            counts[label] += 1
        print(f"  Auto-labelled {len(updates)} rows:")
        print(f"    Low (0):      {counts[0]}")
        print(f"    Moderate (1): {counts[1]}")
        print(f"    High (2):     {counts[2]}\n")

    cursor.close()
    conn.close()

# ===============================
# STEP 3 — IMPORT PAGASA CSV
# Usage:
#   python train_model.py --import pagasa_data.csv
#
# Expected CSV columns (flexible):
#   datetime/timestamp, rainfall/rain,
#   water_level, flow_rate, flooded
# ===============================
def import_pagasa_csv(filepath):
    print(f"📂 Importing PAGASA data from: {filepath}")

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()

    # Flexible column name mapping
    rename = {}
    for col in df.columns:
        if col in ("datetime", "date", "time"):
            rename[col] = "timestamp"
        elif col in ("rainfall", "rain", "rain_mm", "precipitation"):
            rename[col] = "total_rain"
        elif col in ("water_level_cm", "waterlevel", "level"):
            rename[col] = "water_level"
        elif col in ("flow", "flowrate", "flow_lps"):
            rename[col] = "flow_rate"
    df.rename(columns=rename, inplace=True)

    required = ["timestamp", "total_rain", "water_level", "flow_rate"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ CSV is missing columns: {missing}")
        print(f"   Found columns: {list(df.columns)}")
        return

    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["total_rain"]  = pd.to_numeric(df["total_rain"],  errors="coerce").fillna(0)
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce").fillna(0)
    df["flow_rate"]   = pd.to_numeric(df["flow_rate"],   errors="coerce").fillna(0)

    if "flooded" not in df.columns:
        print("  ℹ️  No 'flooded' column in CSV — applying rule-based labels.")
        df["flooded"] = df.apply(
            lambda r: 2 if (r.water_level > 150 or r.total_rain > 30)
                     else (1 if (r.water_level > 80 or r.total_rain > 15 or r.flow_rate > 3)
                     else 0), axis=1
        )

    if "subside_time" not in df.columns:
        df["subside_time"] = 0.0

    df["flooded"]      = pd.to_numeric(df["flooded"],      errors="coerce").fillna(0).astype(int)
    df["subside_time"] = pd.to_numeric(df["subside_time"], errors="coerce").fillna(0)

    conn   = get_db_connection()
    cursor = conn.cursor()

    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(
                """INSERT INTO sensor_readings
                   (timestamp, total_rain, water_level, flow_rate, flooded, subside_time)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    row["timestamp"],
                    row["total_rain"],
                    row["water_level"],
                    row["flow_rate"],
                    int(row["flooded"]),
                    float(row["subside_time"])
                )
            )
            inserted += 1
        except Exception as e:
            print(f"  ⚠️  Row insert error: {e}")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Imported {inserted} rows from PAGASA CSV.\n")

# ===============================
# STEP 4 — LOAD DATA FROM MYSQL
# ===============================
def load_sensor_data():
    try:
        conn = get_db_connection()
        df   = pd.read_sql(
            "SELECT * FROM sensor_readings ORDER BY timestamp ASC", conn
        )
        conn.close()
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        return pd.DataFrame()

# ===============================
# STEP 5 — FEATURE ENGINEERING
# ===============================
def prepare_features(df):
    if df.empty:
        print("❌ No data to train on!")
        return None, None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Rolling rainfall (window = readings, not time)
    df["rain_3h"] = df["total_rain"].rolling(window=3, min_periods=1).sum()
    df["rain_6h"] = df["total_rain"].rolling(window=6, min_periods=1).sum()

    # Water level rise rate
    df["rise_rate"] = df["water_level"].diff().fillna(0)

    # Month
    df["month"] = df["timestamp"].dt.month

    df = df.fillna(0)

    features = ["total_rain", "water_level", "flow_rate",
                "rain_3h", "rain_6h", "rise_rate", "month"]

    # Ensure label columns exist
    if "flooded" not in df.columns:
        df["flooded"] = 0
    if "subside_time" not in df.columns:
        df["subside_time"] = 0

    X       = df[features]
    y_class = df["flooded"].astype(int)
    y_reg   = df["subside_time"].astype(float)

    # ---- GUARD: check class diversity ----
    unique_classes = y_class.nunique()
    print(f"\n📊 Label distribution:\n{y_class.value_counts().sort_index().to_string()}")
    print(f"   Unique classes: {unique_classes}")

    if unique_classes < 2:
        print("\n❌ TRAINING ABORTED: Only one class in labels.")
        print("   The model cannot learn anything useful with all-identical labels.")
        print("   Fix options:")
        print("   1. Run:  python train_model.py --import your_pagasa_data.csv")
        print("   2. Manually update 'flooded' column in sensor_readings table.")
        print("   3. Delete all rows and let auto_label run with more varied sensor data.\n")
        return None, None, None

    return X, y_class, y_reg

# ===============================
# STEP 6 — TRAIN MODELS
# ===============================
def train_models(X, y_class, y_reg):
    # Need at least 10 rows for a meaningful split
    if len(X) < 10:
        print(f"⚠️  Only {len(X)} rows — using all data for training (no test split).")
        X_train, y_class_train, y_reg_train = X, y_class, y_reg
        X_test,  y_class_test,  y_reg_test  = X, y_class, y_reg
    else:
        X_train, X_test, y_class_train, y_class_test = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        _, _, y_reg_train, y_reg_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

    # --- Classification model ---
    print("\n🌲 Training flood risk classifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",   # handles uneven Low/Moderate/High counts
        random_state=42
    )
    clf.fit(X_train, y_class_train)
    y_pred = clf.predict(X_test)
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_class_test, y_pred,
                                target_names=["Low", "Moderate", "High"],
                                zero_division=0))

    # --- Regression model ---
    print("🌲 Training flood subside time regressor...")
    reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    print("===== REGRESSION METRICS =====")
    print("MAE:", round(mean_absolute_error(y_reg_test, y_reg_pred), 4))
    print("R2: ", round(r2_score(y_reg_test, y_reg_pred), 4))

    # --- Feature importances ---
    feature_names = ["total_rain", "water_level", "flow_rate",
                     "rain_3h", "rain_6h", "rise_rate", "month"]
    importances = pd.Series(clf.feature_importances_, index=feature_names)
    print("\n===== FEATURE IMPORTANCES =====")
    print(importances.sort_values(ascending=False).to_string())

    # Save models
    joblib.dump(clf, "model_class.pkl")
    joblib.dump(reg, "model_subside.pkl")
    print("\n✅ Models saved: model_class.pkl, model_subside.pkl")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    # Handle: python train_model.py --import pagasa_data.csv
    if "--import" in sys.argv:
        idx = sys.argv.index("--import")
        if idx + 1 >= len(sys.argv):
            print("❌ Usage: python train_model.py --import <path_to_csv>")
            sys.exit(1)
        csv_path = sys.argv[idx + 1]
        migrate_table()
        import_pagasa_csv(csv_path)

    # Normal training flow
    print("=" * 40)
    print("FloodGuide Model Training")
    print("=" * 40)

    migrate_table()
    auto_label_if_needed()

    print("📥 Loading sensor data from MySQL...")
    df = load_sensor_data()
    print(f"   {len(df)} rows loaded.\n")

    result = prepare_features(df)
    if result[0] is not None:
        X, y_class, y_reg = result
        train_models(X, y_class, y_reg)
    else:
        print("❌ Training could not proceed. See messages above.")