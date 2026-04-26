"""
import_pagasa_and_train.py
────────────────────────────────────────────────────────
Imports Mactan_Daily_Data.csv from DOST-PAGASA into
Clever Cloud and retrains the flood risk model using
real 2020-2024 rainfall data.

Usage:
    python import_pagasa_and_train.py

Place Mactan_Daily_Data.csv in the same folder as this
script before running.
────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import joblib
import mysql.connector
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# =========================================
# CONFIG
# =========================================
CSV_PATH   = "Mactan_Daily_Data.csv"
MODEL_PATH = "model_class.pkl"

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
# STEP 1 — LOAD AND CLEAN PAGASA CSV
# =========================================
def load_pagasa_csv():
    print("📂 Loading PAGASA data from:", CSV_PATH)

    if not os.path.exists(CSV_PATH):
        print(f"❌ File not found: {CSV_PATH}")
        print("   Make sure Mactan_Daily_Data.csv is in the same folder.")
        return None

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.upper()

    print(f"   Raw rows: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Years: {sorted(df['YEAR'].unique().tolist())}")

    # Convert RAINFALL — handle special values
    df['RAINFALL'] = pd.to_numeric(df['RAINFALL'], errors='coerce')
    df['RAINFALL'] = df['RAINFALL'].apply(
        lambda x: 0.05 if x == -1.0    # -1 = Trace (<0.1mm) → use 0.05
        else (np.nan if x <= -999.0    # -999 = Missing → NaN
        else x)
    )
    df['RAINFALL'] = df['RAINFALL'].fillna(0).clip(lower=0)

    # Build timestamp
    df['timestamp'] = pd.to_datetime(
        df[['YEAR', 'MONTH', 'DAY']].rename(
            columns={'YEAR': 'year', 'MONTH': 'month', 'DAY': 'day'}
        )
    )

    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"   Valid rows after cleaning: {len(df)}")
    print(f"   Date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df

# =========================================
# STEP 2 — ENGINEER FEATURES AND LABELS
# =========================================
def prepare_pagasa_features(df):
    print("\n🔧 Engineering features...")

    # Rolling rainfall sums (daily data, so window=3 = 3-day sum)
    df['rain_3h']  = df['RAINFALL'].rolling(window=3, min_periods=1).sum()
    df['rise_rate'] = 0.0   # PAGASA has no water level — use 0
    df['water_level'] = 0.0
    df['flow_rate']   = 0.0
    df['month']       = df['MONTH']
    df['subside_time'] = 0.0

    # ── FLOOD LABELING ──
    # Based on PAGASA rainfall warning thresholds:
    # Yellow:  >7.5mm/hr  → 15mm daily moderate alert
    # Orange:  >15mm/hr   → 30mm daily high alert
    # Red:     >30mm/hr   → extreme rainfall
    # Source: PAGASA Rainfall Warning System
    def label_flood(row):
        rain = row['RAINFALL']
        r3   = row['rain_3h']

        # High risk — extreme/heavy rain or sustained 3-day accumulation
        if rain > 30 or r3 > 60:
            return 2

        # Moderate risk — significant rain or moderate 3-day accumulation
        elif rain > 15 or r3 > 30:
            return 1

        # Low risk
        else:
            return 0

    df['flooded'] = df.apply(label_flood, axis=1)

    # Print distribution
    counts = df['flooded'].value_counts().sort_index()
    total  = len(df)
    print(f"\n📊 Label distribution (from real PAGASA thresholds):")
    print(f"   Low  (0): {counts.get(0,0):4d} days ({counts.get(0,0)/total*100:.1f}%)")
    print(f"   Mod  (1): {counts.get(1,0):4d} days ({counts.get(1,0)/total*100:.1f}%)")
    print(f"   High (2): {counts.get(2,0):4d} days ({counts.get(2,0)/total*100:.1f}%)")
    print(f"   Total:    {total} days")

    return df

# =========================================
# STEP 3 — IMPORT INTO DATABASE
# =========================================
def import_to_db(df):
    print("\n💾 Importing PAGASA data into Clever Cloud database...")

    try:
        conn   = get_db_connection()
        cursor = conn.cursor()

        # Add device_id column if missing
        try:
            cursor.execute(
                "ALTER TABLE sensor_readings ADD COLUMN device_id VARCHAR(20) NOT NULL DEFAULT 'bridge1'"
            )
            conn.commit()
            print("   ✅ device_id column added")
        except mysql.connector.Error as e:
            if e.errno == 1060:
                print("   ℹ️  device_id column already exists")
            else:
                print(f"   ⚠️  Column warning: {e}")

        # Check how many PAGASA rows already exist
        cursor.execute(
            "SELECT COUNT(*) FROM sensor_readings WHERE device_id = 'pagasa'"
        )
        existing = cursor.fetchone()[0]

        if existing > 0:
            print(f"   ℹ️  {existing} PAGASA rows already in DB — skipping import.")
            print("      Delete them first if you want to reimport.")
            cursor.close()
            conn.close()
            return True

        # Insert all rows
        inserted = 0
        errors   = 0

        for _, row in df.iterrows():
            try:
                cursor.execute(
                    """INSERT INTO sensor_readings
                       (device_id, timestamp, total_rain, water_level,
                        flow_rate, flooded, subside_time)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        'pagasa',                    # mark as PAGASA data
                        row['timestamp'],
                        float(row['RAINFALL']),
                        float(row['water_level']),   # 0.0 — not in PAGASA data
                        float(row['flow_rate']),     # 0.0 — not in PAGASA data
                        int(row['flooded']),
                        float(row['subside_time'])
                    )
                )
                inserted += 1
            except Exception as e:
                errors += 1

        conn.commit()
        cursor.close()
        conn.close()

        print(f"   ✅ Inserted {inserted} rows | Errors: {errors}")
        return True

    except Exception as e:
        print(f"   ❌ DB connection failed: {e}")
        print("   ⚠️  Training locally only (model will still be saved)")
        return False

# =========================================
# STEP 4 — TRAIN MODEL
# =========================================
def train_model(df):
    print("\n🌲 Training flood risk model on PAGASA data...")

    features = ['RAINFALL', 'water_level', 'flow_rate',
                'rain_3h', 'rise_rate', 'month']

    # Rename RAINFALL to match the model's expected feature name
    df = df.rename(columns={'RAINFALL': 'total_rain'})
    features = ['total_rain', 'water_level', 'flow_rate',
                'rain_3h', 'rise_rate', 'month']

    X = df[features]
    y = df['flooded'].astype(int)

    # Stratified split to keep class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",   # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(
        y_test, y_pred,
        target_names=["Low", "Moderate", "High"],
        zero_division=0
    ))

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=features)
    print("===== FEATURE IMPORTANCES =====")
    print(importances.sort_values(ascending=False).to_string())

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"   Classes: {clf.classes_.tolist()}")
    print(f"   Trained on: {len(X_train)} samples | Tested on: {len(X_test)} samples")

    return clf

# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    print("=" * 50)
    print("FloodGuide — PAGASA Data Import & Model Training")
    print("=" * 50)

    # Step 1
    df = load_pagasa_csv()
    if df is None:
        exit(1)

    # Step 2
    df = prepare_pagasa_features(df)

    # Step 3 — import to DB (optional — continues even if DB fails)
    import_to_db(df)

    # Step 4 — train model
    clf = train_model(df)

    print("\n" + "=" * 50)
    print("DONE! Next steps:")
    print("1. Upload model_class.pkl to your GitHub repo")
    print("2. Render will auto-deploy with the new model")
    print("3. Visit /retrain to verify the endpoint works")
    print("=" * 50)