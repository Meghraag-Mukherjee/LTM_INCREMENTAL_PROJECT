import csv
import json
import os
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras


MODEL_PATH = (
    "models/demand_seq2seq_attention.keras"
)

CONFIG_PATH = "models/seq2seq_config.json"


# ============================================================
# 1. LOAD MODEL AND CONFIG
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run python -m "
        "src.train_seq2seq_attention first."
    )

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"Configuration not found: "
        f"{CONFIG_PATH}"
    )

model = keras.models.load_model(
    MODEL_PATH
)

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)

STORE_ID = int(config["store_id"])
ITEM_ID = int(config["item_id"])

LOOKBACK = int(config["lookback"])

FORECAST_HORIZON = int(
    config["forecast_horizon"]
)

SALES_MINIMUM = float(
    config["sales_minimum"]
)

SALES_RANGE = float(
    config["sales_range"]
)

DATA_PATH = config["data_path"]


# ============================================================
# 2. LOAD LATEST STORE-ITEM SALES
# ============================================================

sales_by_date = {}

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        try:
            store = int(row["store"])
            item = int(row["item"])
            sale = float(row["sales"])

            date = datetime.strptime(
                row["date"],
                "%Y-%m-%d"
            ).date()

        except (ValueError, TypeError):
            continue

        if (
            store == STORE_ID
            and item == ITEM_ID
        ):
            sales_by_date[date] = (
                sales_by_date.get(date, 0.0)
                + sale
            )

if not sales_by_date:
    raise ValueError(
        "No matching store-item data found."
    )


# ============================================================
# 3. CREATE COMPLETE DAILY SERIES
# ============================================================

start_date = min(sales_by_date)
last_date = max(sales_by_date)

sales = []

current_date = start_date

while current_date <= last_date:

    sales.append(
        float(
            sales_by_date.get(
                current_date,
                0.0
            )
        )
    )

    current_date += timedelta(days=1)


# ============================================================
# 4. MULTI-HORIZON FORECAST
# ============================================================

def forecast_next_horizon():

    if len(sales) < LOOKBACK:
        raise ValueError(
            f"At least {LOOKBACK} historical "
            "days are required."
        )

    latest_sales = sales[-LOOKBACK:]

    scaled_sales = [
        (
            value - SALES_MINIMUM
        ) / SALES_RANGE
        for value in latest_sales
    ]

    input_values = [
        [[value]]
        for value in scaled_sales
    ]

    input_tensor = tf.constant(
        input_values,
        dtype=tf.float32
    )

    input_tensor = tf.reshape(
        input_tensor,
        shape=(
            1,
            LOOKBACK,
            1
        )
    )

    scaled_predictions = model.predict(
        input_tensor,
        verbose=0
    )[0, :, 0]

    forecasts = []

    for horizon_index in range(
        FORECAST_HORIZON
    ):

        predicted_sales = (
            float(
                scaled_predictions[
                    horizon_index
                ]
            )
            * SALES_RANGE
            + SALES_MINIMUM
        )

        predicted_sales = max(
            0.0,
            predicted_sales
        )

        forecast_date = (
            last_date
            + timedelta(
                days=horizon_index + 1
            )
        )

        forecasts.append({
            "horizon_day": horizon_index + 1,

            "date": forecast_date.isoformat(),

            "predicted_sales": round(
                predicted_sales,
                2
            )
        })

    return {
        "store": STORE_ID,
        "item": ITEM_ID,
        "lookback_days": LOOKBACK,
        "forecast_horizon": (
            FORECAST_HORIZON
        ),
        "last_observed_date": (
            last_date.isoformat()
        ),
        "forecasts": forecasts
    }


# ============================================================
# 5. RUN
# ============================================================

if __name__ == "__main__":

    result = forecast_next_horizon()

    print("\n" + "=" * 60)
    print("MULTI-HORIZON DEMAND FORECAST")
    print("=" * 60)

    print("Store:", result["store"])
    print("Item :", result["item"])

    print(
        "Last observed date:",
        result["last_observed_date"]
    )

    print(
        "Forecast horizon:",
        result["forecast_horizon"],
        "days"
    )

    print("\nForecasts:")

    for forecast in result["forecasts"]:

        print(
            f"Day +{forecast['horizon_day']} | "
            f"{forecast['date']} | "
            f"Predicted sales: "
            f"{forecast['predicted_sales']}"
        )