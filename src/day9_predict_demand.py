import json
import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras


DATA_PATH = "data/store_item_demand/train.csv"

MODEL_PATH = "models/demand_lstm.keras"

SCALER_PATH = "models/demand_scaler.json"


# ============================================================
# 1. CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run python -m src.train_demand_lstm first."
    )

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"Scaling file not found: {SCALER_PATH}"
    )


# ============================================================
# 2. LOAD MODEL AND SCALER INFORMATION
# ============================================================

model = keras.models.load_model(
    MODEL_PATH
)

with open(
    SCALER_PATH,
    "r",
    encoding="utf-8"
) as file:
    scaler = json.load(file)

sales_minimum = scaler[
    "sales_minimum"
]

sales_range = scaler[
    "sales_range"
]

lookback = int(
    scaler["lookback"]
)

store_id = int(
    scaler["store_id"]
)

item_id = int(
    scaler["item_id"]
)


# ============================================================
# 3. LOAD LATEST SALES
# ============================================================

data = pd.read_csv(
    DATA_PATH
)

data["date"] = pd.to_datetime(
    data["date"],
    errors="coerce"
)

data["sales"] = pd.to_numeric(
    data["sales"],
    errors="coerce"
)

data = data.dropna(
    subset=[
        "date",
        "store",
        "item",
        "sales"
    ]
)

series_data = data[
    (data["store"] == store_id)
    &
    (data["item"] == item_id)
].copy()

daily_sales = (
    series_data
    .set_index("date")["sales"]
    .resample("D")
    .sum()
    .asfreq("D", fill_value=0)
    .sort_index()
)


# ============================================================
# 4. NEXT-DAY PREDICTION
# ============================================================

def predict_next_day():
    """
    Predict demand for the day immediately after
    the latest available date.
    """

    if len(daily_sales) < lookback:
        raise ValueError(
            f"At least {lookback} daily observations "
            "are required."
        )

    latest_window = daily_sales.tail(
        lookback
    )

    scaled_window = (
        latest_window -
        sales_minimum
    ) / sales_range

    input_tensor = tf.constant(
        scaled_window.tolist(),
        dtype=tf.float32
    )

    input_tensor = tf.reshape(
        input_tensor,
        shape=(
            1,
            lookback,
            1
        )
    )

    scaled_prediction = float(
        model.predict(
            input_tensor,
            verbose=0
        )[0][0]
    )

    predicted_sales = (
        scaled_prediction *
        sales_range
        +
        sales_minimum
    )

    predicted_sales = max(
        0.0,
        predicted_sales
    )

    next_date = (
        daily_sales.index.max()
        +
        pd.Timedelta(days=1)
    )

    return {
        "store": store_id,
        "item": item_id,
        "forecast_date": str(
            next_date.date()
        ),
        "predicted_sales": round(
            predicted_sales,
            2
        ),
        "lookback_days": lookback
    }


# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":

    result = predict_next_day()

    print("\nNext-day demand forecast:")

    print(
        "Store:",
        result["store"]
    )

    print(
        "Item:",
        result["item"]
    )

    print(
        "Forecast date:",
        result["forecast_date"]
    )

    print(
        "Predicted sales:",
        result["predicted_sales"]
    )

    print(
        "Look-back days:",
        result["lookback_days"]
    )
