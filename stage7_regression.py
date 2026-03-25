import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib

PROJECT_ROOT = Path.cwd()
DATA_PATH = PROJECT_ROOT / "outputs/features/stage6_with_gt.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset size:", len(df))

# -------------------------------------------------
# ORIGINAL FEATURES (BEST)
# -------------------------------------------------

FEATURES = [
    "rel_area_loss",
    "convexity_dev",
    "mean_embedding_distance",
    "rel_perimeter_change",
    "compactness_dev"
]

TARGET = "defoliation_gt"

X = df[FEATURES]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))

# -------------------------------------------------
# MODEL
# -------------------------------------------------

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------

y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\nValidation Results")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2  :", round(r2, 3))

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

MODEL_PATH = PROJECT_ROOT / "outputs/model/defoliation_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("Model saved:", MODEL_PATH)