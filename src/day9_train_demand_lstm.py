import json
import os
import random

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/store_item_demand/train.csv"

MODEL_PATH = "models/demand_lstm.keras"

SCALER_PATH = "models/demand_scaler.json"

FORECAST_PATH = "results/demand_forecast.csv"

METRICS_PATH = "results/demand_metrics.csv"

HISTORY_PATH = (
    "results/demand_training_history.csv"
)

PLOT_PATH = "results/demand_forecast.png"

RANDOM_STATE = 42

# Select one store-product series.
STORE_ID = 1
ITEM_ID = 1

# Use the previous 30 days to predict the next day.
LOOKBACK = 30

# Most recent 90 days become the test period.
TEST_DAYS = 90

VALIDATION_DAYS = 60

MAX_EPOCHS = 50

BATCH_SIZE = 32


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 4. LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place the Store Item Demand train.csv "
        "inside data/store_item_demand/."
    )

data = pd.read_csv(
    DATA_PATH
)

print("=" * 70)
print("DAY 9: LSTM DEMAND FORECASTING")
print("=" * 70)

print("Original dataset shape:", data.shape)

print("\nColumns:")

print(
    data.columns.tolist()
)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "date",
    "store",
    "item",
    "sales"
}

missing_columns = (
    required_columns -
    set(data.columns)
)

if missing_columns:
    raise ValueError(
        "Dataset is missing columns: "
        f"{missing_columns}"
    )


# ============================================================
# 6. CLEAN DATA
# ============================================================

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

data = data.sort_values(
    "date"
)

print(
    "\nDate range:",
    data["date"].min(),
    "to",
    data["date"].max()
)


# ============================================================
# 7. SELECT ONE STORE AND ITEM
# ============================================================

series_data = data[
    (data["store"] == STORE_ID)
    &
    (data["item"] == ITEM_ID)
].copy()

if series_data.empty:
    raise ValueError(
        f"No data found for store {STORE_ID} "
        f"and item {ITEM_ID}."
    )

print(
    f"\nSelected store: {STORE_ID}"
)

print(
    f"Selected item : {ITEM_ID}"
)

print(
    "Selected rows:",
    len(series_data)
)


# ============================================================
# 8. CREATE CLEAN DAILY TIME SERIES
# ============================================================

daily_sales = (
    series_data
    .set_index("date")["sales"]
    .resample("D")
    .sum()
    .asfreq("D", fill_value=0)
)

daily_sales = daily_sales.astype(
    "float32"
)

print(
    "\nDaily observations:",
    len(daily_sales)
)

if len(daily_sales) <= (
    LOOKBACK
    + TEST_DAYS
    + VALIDATION_DAYS
):
    raise ValueError(
        "Not enough observations for the selected "
        "look-back, validation and test periods."
    )


# ============================================================
# 9. CHRONOLOGICAL SPLIT
# ============================================================

test_start_position = (
    len(daily_sales) -
    TEST_DAYS
)

validation_start_position = (
    test_start_position -
    VALIDATION_DAYS
)

training_series = daily_sales.iloc[
    :validation_start_position
]

validation_series = daily_sales.iloc[
    validation_start_position:
    test_start_position
]

test_series = daily_sales.iloc[
    test_start_position:
]

print("\nChronological split:")

print(
    "Training observations  :",
    len(training_series)
)

print(
    "Validation observations:",
    len(validation_series)
)

print(
    "Testing observations   :",
    len(test_series)
)

print(
    "\nTraining period:",
    training_series.index.min(),
    "to",
    training_series.index.max()
)

print(
    "Validation period:",
    validation_series.index.min(),
    "to",
    validation_series.index.max()
)

print(
    "Testing period:",
    test_series.index.min(),
    "to",
    test_series.index.max()
)


# ============================================================
# 10. MIN-MAX SCALING
# ============================================================

# Fit scaling values only on the training period.
sales_minimum = float(
    training_series.min()
)

sales_maximum = float(
    training_series.max()
)

sales_range = (
    sales_maximum -
    sales_minimum
)

if sales_range == 0:
    raise ValueError(
        "Training sales are constant, so scaling "
        "and forecasting cannot be performed."
    )


def scale_sales(series):
    """
    Scale sales to approximately 0-1 based only
    on the training-period minimum and maximum.
    """

    return (
        series -
        sales_minimum
    ) / sales_range


def inverse_scale(values):
    """
    Convert scaled model output back into sales units.
    """

    return (
        values * sales_range
        +
        sales_minimum
    )


scaled_daily_sales = scale_sales(
    daily_sales
)


# ============================================================
# 11. SAVE SCALING INFORMATION
# ============================================================

scaler_information = {
    "sales_minimum": sales_minimum,
    "sales_maximum": sales_maximum,
    "sales_range": sales_range,
    "lookback": LOOKBACK,
    "store_id": STORE_ID,
    "item_id": ITEM_ID
}

with open(
    SCALER_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        scaler_information,
        file,
        indent=4
    )


# ============================================================
# 12. CREATE LOOK-BACK WINDOWS
# ============================================================

def create_windows(
    scaled_series,
    lookback
):
    """
    Convert a time series into supervised samples.

    Previous lookback days -> next day's sales
    """

    feature_windows = []
    targets = []
    target_dates = []

    for target_position in range(
        lookback,
        len(scaled_series)
    ):
        start_position = (
            target_position -
            lookback
        )

        window = scaled_series.iloc[
            start_position:
            target_position
        ].tolist()

        target = scaled_series.iloc[
            target_position
        ]

        target_date = scaled_series.index[
            target_position
        ]

        feature_windows.append(
            window
        )

        targets.append(
            float(target)
        )

        target_dates.append(
            target_date
        )

    features = tf.constant(
        feature_windows,
        dtype=tf.float32
    )

    # LSTM requires:
    # samples x timesteps x features
    features = tf.expand_dims(
        features,
        axis=-1
    )

    targets = tf.constant(
        targets,
        dtype=tf.float32
    )

    return (
        features,
        targets,
        target_dates
    )


all_X, all_y, all_target_dates = (
    create_windows(
        scaled_daily_sales,
        LOOKBACK
    )
)

print(
    "\nComplete LSTM input shape:",
    all_X.shape
)

print(
    "Complete target shape:",
    all_y.shape
)


# ============================================================
# 13. SPLIT WINDOWS BY TARGET DATE
# ============================================================

training_end_date = (
    training_series.index.max()
)

validation_end_date = (
    validation_series.index.max()
)

train_indices = [
    index
    for index, date
    in enumerate(all_target_dates)
    if date <= training_end_date
]

validation_indices = [
    index
    for index, date
    in enumerate(all_target_dates)
    if (
        date > training_end_date
        and
        date <= validation_end_date
    )
]

test_indices = [
    index
    for index, date
    in enumerate(all_target_dates)
    if date > validation_end_date
]

X_train = tf.gather(
    all_X,
    train_indices
)

y_train = tf.gather(
    all_y,
    train_indices
)

X_validation = tf.gather(
    all_X,
    validation_indices
)

y_validation = tf.gather(
    all_y,
    validation_indices
)

X_test = tf.gather(
    all_X,
    test_indices
)

y_test = tf.gather(
    all_y,
    test_indices
)

test_dates = [
    all_target_dates[index]
    for index in test_indices
]

print("\nWindow shapes:")

print(
    "X_train     :",
    X_train.shape
)

print(
    "X_validation:",
    X_validation.shape
)

print(
    "X_test      :",
    X_test.shape
)


# ============================================================
# 14. BUILD LSTM MODEL
# ============================================================

model = keras.Sequential(
    [
        layers.Input(
            shape=(
                LOOKBACK,
                1
            ),
            name="sales_window"
        ),

        layers.LSTM(
            units=64,
            return_sequences=True,
            name="lstm_1"
        ),

        layers.Dropout(
            rate=0.20,
            name="dropout_1"
        ),

        layers.LSTM(
            units=32,
            name="lstm_2"
        ),

        layers.Dropout(
            rate=0.20,
            name="dropout_2"
        ),

        layers.Dense(
            units=16,
            activation="relu",
            name="dense_1"
        ),

        # Regression output: one next-day sales value.
        layers.Dense(
            units=1,
            name="next_day_sales"
        )
    ],
    name="retailmind_demand_lstm"
)


# ============================================================
# 15. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mean_squared_error",

    metrics=[
        keras.metrics.MeanAbsoluteError(
            name="mae"
        ),

        keras.metrics.RootMeanSquaredError(
            name="rmse"
        )
    ]
)

print("\nLSTM architecture:")

model.summary()


# ============================================================
# 16. CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    mode="min",
    restore_best_weights=True,
    verbose=1
)

reduce_learning_rate = (
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
)

model_checkpoint = (
    keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )
)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LSTM")
print("=" * 70)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=MAX_EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=False,

    callbacks=[
        early_stopping,
        reduce_learning_rate,
        model_checkpoint
    ],

    verbose=1
)


# ============================================================
# 18. LOAD BEST SAVED MODEL
# ============================================================

best_model = keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# 19. PREDICT TEST PERIOD
# ============================================================

scaled_predictions = (
    best_model.predict(
        X_test,
        verbose=0
    )
    .reshape(-1)
)

scaled_actual_values = (
    y_test.numpy()
    .reshape(-1)
)

predicted_sales = inverse_scale(
    scaled_predictions
)

actual_sales = inverse_scale(
    scaled_actual_values
)

# Sales cannot normally be negative.
predicted_sales = tf.maximum(
    predicted_sales,
    0
).numpy()


# ============================================================
# 20. CALCULATE MAE AND RMSE
# ============================================================

actual_tensor = tf.constant(
    actual_sales,
    dtype=tf.float32
)

prediction_tensor = tf.constant(
    predicted_sales,
    dtype=tf.float32
)

mae = float(
    tf.reduce_mean(
        tf.abs(
            actual_tensor -
            prediction_tensor
        )
    ).numpy()
)

rmse = float(
    tf.sqrt(
        tf.reduce_mean(
            tf.square(
                actual_tensor -
                prediction_tensor
            )
        )
    ).numpy()
)

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# ============================================================
# 21. SAVE FORECAST RESULTS
# ============================================================

forecast_results = pd.DataFrame({
    "date": test_dates,
    "actual_sales": actual_sales,
    "predicted_sales": predicted_sales
})

forecast_results[
    "absolute_error"
] = (
    forecast_results["actual_sales"]
    -
    forecast_results["predicted_sales"]
).abs()

forecast_results.to_csv(
    FORECAST_PATH,
    index=False
)

metrics = pd.DataFrame([
    {
        "store": STORE_ID,
        "item": ITEM_ID,
        "lookback_days": LOOKBACK,
        "test_days": TEST_DAYS,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4)
    }
])

metrics.to_csv(
    METRICS_PATH,
    index=False
)

pd.DataFrame(
    history.history
).to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# 22. PLOT ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    forecast_results["date"],
    forecast_results["actual_sales"],
    label="Actual Sales",
    color="blue",
    linewidth=2
)

plt.plot(
    forecast_results["date"],
    forecast_results["predicted_sales"],
    label="Predicted Sales",
    color="orange",
    linewidth=2
)

plt.title(
    f"Demand Forecast: Store {STORE_ID}, "
    f"Item {ITEM_ID}"
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(alpha=0.25)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150
)

plt.close()


# ============================================================
# 23. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("LSTM model      :", MODEL_PATH)
print("Scaling details :", SCALER_PATH)
print("Forecast values :", FORECAST_PATH)
print("Metrics         :", METRICS_PATH)
print("Training history:", HISTORY_PATH)
print("Forecast plot   :", PLOT_PATH)
