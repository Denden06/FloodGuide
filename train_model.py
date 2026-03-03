import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("historical_flood_data.csv")

# Ensure correct datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ===============================
# FEATURE ENGINEERING
# ===============================

# Rolling rainfall accumulation (last 3 hours)
df["rain_3h"] = df["rainfall"].rolling(window=3).sum()

# Rolling rainfall accumulation (last 6 hours)
df["rain_6h"] = df["rainfall"].rolling(window=6).sum()

# Water level rise rate
df["rise_rate"] = df["water_level"].diff()

# Seasonal feature (month)
df["month"] = df["timestamp"].dt.month

# Fill missing values
df = df.fillna(0)

# ===============================
# FEATURES & TARGETS
# ===============================

features = [
    "rainfall",
    "water_level",
    "flow_rate",
    "rain_3h",
    "rain_6h",
    "rise_rate",
    "month"
]

X = df[features]
y_class = df["flooded"]
y_reg = df["subside_time"]

# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# ===============================
# CLASSIFICATION MODEL
# ===============================

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

clf.fit(X_train, y_class_train)

y_class_pred = clf.predict(X_test)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_class_test, y_class_pred))

# ===============================
# REGRESSION MODEL
# ===============================

reg = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

reg.fit(X_train, y_reg_train)

y_reg_pred = reg.predict(X_test)

print("\n===== REGRESSION METRICS =====")
print("MAE:", mean_absolute_error(y_reg_test, y_reg_pred))
print("R2:", r2_score(y_reg_test, y_reg_pred))

# ===============================
# SAVE MODELS
# ===============================

joblib.dump(clf, "model_class.pkl")
joblib.dump(reg, "model_subside.pkl")

print("\nModels saved successfully.")