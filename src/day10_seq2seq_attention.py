import csv
import json
import math
import os
import random
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/store_item_demand/train.csv"

MODEL_PATH = (
    "models/demand_seq2seq_attention.keras"
)

CONFIG_PATH = "models/seq2seq_config.json"

FORECAST_PATH = "results/seq2seq_forecast.csv"

METRICS_PATH = "results/seq2seq_metrics.csv"

HORIZON_METRICS_PATH = (
    "results/horizon_metrics.csv"
)

ATTENTION_PATH = (
    "results/attention_weights.csv"
)

ATTENTION_IMAGE_PATH = (
    "results/attention_heatmap.png"
)

HISTORY_PATH = (
    "results/seq2seq_training_history.csv"
)

RANDOM_STATE = 42

STORE_ID = 1
ITEM_ID = 1

# Previous 30 days
LOOKBACK = 30

# Predict next 7 days
FORECAST_HORIZON = 7

# Most recent 90 forecast windows for testing
TEST_WINDOWS = 90

# Previous 60 windows for validation
VALIDATION_WINDOWS = 60

ENCODER_UNITS = 64
DECODER_UNITS = 64

MAX_EPOCHS = 50
BATCH_SIZE = 32


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. CHECK FILES AND DIRECTORIES
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place train.csv inside "
        "data/store_item_demand/."
    )

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 4. LOAD ONE STORE-ITEM SERIES
# ============================================================

sales_by_date = {}

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    required_columns = {
        "date",
        "store",
        "item",
        "sales"
    }

    if not required_columns.issubset(
        reader.fieldnames
    ):
        raise ValueError(
            "CSV must contain date, store, "
            "item and sales columns."
        )

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
        f"No data found for store {STORE_ID}, "
        f"item {ITEM_ID}."
    )


# ============================================================
# 5. BUILD COMPLETE DAILY SERIES
# ============================================================

start_date = min(sales_by_date)
end_date = max(sales_by_date)

dates = []
sales = []

current_date = start_date

while current_date <= end_date:

    dates.append(current_date)

    sales.append(
        float(
            sales_by_date.get(
                current_date,
                0.0
            )
        )
    )

    current_date += timedelta(days=1)

print("=" * 70)
print("DAY 10: SEQ2SEQ + ATTENTION FORECASTING")
print("=" * 70)

print("Store:", STORE_ID)
print("Item :", ITEM_ID)
print("First date:", start_date)
print("Last date :", end_date)
print("Daily observations:", len(sales))
print("Look-back days:", LOOKBACK)
print("Forecast horizon:", FORECAST_HORIZON)


# ============================================================
# 6. CHECK DATA SIZE
# ============================================================

minimum_required = (
    LOOKBACK
    + FORECAST_HORIZON
    + TEST_WINDOWS
    + VALIDATION_WINDOWS
)

if len(sales) < minimum_required:
    raise ValueError(
        "Not enough data. At least "
        f"{minimum_required} daily observations "
        "are required."
    )


# ============================================================
# 7. TRAIN-ONLY SCALING
# ============================================================

# Validation/test boundaries are defined by target windows.
test_start_target = (
    len(sales)
    - TEST_WINDOWS
    - FORECAST_HORIZON
    + 1
)

validation_start_target = (
    test_start_target
    - VALIDATION_WINDOWS
)

# Scaling is fitted only on historical values available
# before validation and test targets.
training_values = sales[
    :validation_start_target
]

sales_minimum = min(training_values)
sales_maximum = max(training_values)

sales_range = (
    sales_maximum
    - sales_minimum
)

if sales_range == 0:
    raise ValueError(
        "Training sales are constant."
    )


def scale_value(value):

    return (
        value - sales_minimum
    ) / sales_range


def inverse_scale(value):

    return (
        value * sales_range
        + sales_minimum
    )


scaled_sales = [
    scale_value(value)
    for value in sales
]


# ============================================================
# 8. CREATE SEQ2SEQ WINDOWS
# ============================================================

feature_windows = []
target_windows = []
target_date_windows = []

for target_start in range(
    LOOKBACK,
    len(sales) - FORECAST_HORIZON + 1
):

    history_start = (
        target_start - LOOKBACK
    )

    history_window = scaled_sales[
        history_start:target_start
    ]

    future_window = scaled_sales[
        target_start:
        target_start + FORECAST_HORIZON
    ]

    future_dates = dates[
        target_start:
        target_start + FORECAST_HORIZON
    ]

    feature_windows.append(
        [
            [value]
            for value in history_window
        ]
    )

    target_windows.append(
        [
            [value]
            for value in future_window
        ]
    )

    target_date_windows.append(
        future_dates
    )

X = tf.constant(
    feature_windows,
    dtype=tf.float32
)

y = tf.constant(
    target_windows,
    dtype=tf.float32
)

print("\nComplete feature shape:", X.shape)
print("Complete target shape :", y.shape)


# ============================================================
# 9. CHRONOLOGICAL WINDOW SPLIT
# ============================================================

total_windows = int(X.shape[0])

test_start_index = (
    total_windows - TEST_WINDOWS
)

validation_start_index = (
    test_start_index
    - VALIDATION_WINDOWS
)

X_train = X[:validation_start_index]
y_train = y[:validation_start_index]

X_validation = X[
    validation_start_index:
    test_start_index
]

y_validation = y[
    validation_start_index:
    test_start_index
]

X_test = X[test_start_index:]
y_test = y[test_start_index:]

test_target_dates = target_date_windows[
    test_start_index:
]

print("\nChronological split:")

print("Training windows  :", X_train.shape[0])
print("Validation windows:", X_validation.shape[0])
print("Testing windows   :", X_test.shape[0])


# ============================================================
# 10. BUILD SEQ2SEQ + ATTENTION MODEL
# ============================================================

encoder_inputs = keras.Input(
    shape=(LOOKBACK, 1),
    name="history_window"
)

# Encoder returns:
# 1. output for every historical step
# 2. final hidden state
# 3. final cell state
encoder_lstm = layers.LSTM(
    ENCODER_UNITS,
    return_sequences=True,
    return_state=True,
    name="encoder_lstm"
)

(
    encoder_outputs,
    encoder_hidden_state,
    encoder_cell_state
) = encoder_lstm(
    encoder_inputs
)

# Repeat final encoder state once for each
# future step required.
decoder_seed = layers.RepeatVector(
    FORECAST_HORIZON,
    name="repeat_encoder_state"
)(
    encoder_hidden_state
)

decoder_lstm = layers.LSTM(
    DECODER_UNITS,
    return_sequences=True,
    name="decoder_lstm"
)

decoder_outputs = decoder_lstm(
    decoder_seed,
    initial_state=[
        encoder_hidden_state,
        encoder_cell_state
    ]
)

# AdditiveAttention compares each decoder step with
# all encoder history steps.
attention_layer = layers.AdditiveAttention(
    name="history_attention"
)

(
    attention_context,
    attention_scores
) = attention_layer(
    [
        decoder_outputs,
        encoder_outputs
    ],
    return_attention_scores=True
)

combined_context = layers.Concatenate(
    name="decoder_attention_context"
)(
    [
        decoder_outputs,
        attention_context
    ]
)

hidden_outputs = layers.TimeDistributed(
    layers.Dense(
        32,
        activation="relu"
    ),
    name="forecast_hidden"
)(
    combined_context
)

forecast_outputs = layers.TimeDistributed(
    layers.Dense(1),
    name="multi_horizon_sales"
)(
    hidden_outputs
)

forecast_model = keras.Model(
    inputs=encoder_inputs,
    outputs=forecast_outputs,
    name="retailmind_seq2seq_attention"
)

# Separate model for attention visualization.
attention_model = keras.Model(
    inputs=encoder_inputs,
    outputs=attention_scores,
    name="retailmind_attention_inspector"
)


# ============================================================
# 11. COMPILE MODEL
# ============================================================

forecast_model.compile(
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

forecast_model.summary()


# ============================================================
# 12. CALLBACKS
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

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 13. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SEQ2SEQ + ATTENTION")
print("=" * 70)

history = forecast_model.fit(
    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,

    # Time-series windows remain in chronological order.
    shuffle=False,

    callbacks=[
        early_stopping,
        reduce_learning_rate,
        checkpoint
    ],

    verbose=1
)


# ============================================================
# 14. LOAD BEST MODEL
# ============================================================

best_model = keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# 15. TEST PREDICTIONS
# ============================================================

scaled_predictions = best_model.predict(
    X_test,
    verbose=0
)

scaled_actual = y_test.numpy()

actual_values = []
predicted_values = []

for sample_index in range(TEST_WINDOWS):

    actual_sample = []
    predicted_sample = []

    for horizon_index in range(
        FORECAST_HORIZON
    ):

        actual_sale = inverse_scale(
            float(
                scaled_actual[
                    sample_index,
                    horizon_index,
                    0
                ]
            )
        )

        predicted_sale = inverse_scale(
            float(
                scaled_predictions[
                    sample_index,
                    horizon_index,
                    0
                ]
            )
        )

        predicted_sale = max(
            0.0,
            predicted_sale
        )

        actual_sample.append(actual_sale)
        predicted_sample.append(predicted_sale)

    actual_values.append(actual_sample)
    predicted_values.append(predicted_sample)


# ============================================================
# 16. OVERALL MAE AND RMSE
# ============================================================

absolute_errors = []
squared_errors = []

for sample_index in range(TEST_WINDOWS):

    for horizon_index in range(
        FORECAST_HORIZON
    ):

        error = (
            actual_values[sample_index][horizon_index]
            -
            predicted_values[sample_index][horizon_index]
        )

        absolute_errors.append(
            abs(error)
        )

        squared_errors.append(
            error ** 2
        )

overall_mae = (
    sum(absolute_errors)
    / len(absolute_errors)
)

overall_rmse = math.sqrt(
    sum(squared_errors)
    / len(squared_errors)
)

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"Overall MAE : {overall_mae:.4f}")
print(f"Overall RMSE: {overall_rmse:.4f}")


# ============================================================
# 17. PER-HORIZON METRICS
# ============================================================

horizon_metrics = []

for horizon_index in range(
    FORECAST_HORIZON
):

    horizon_absolute_errors = []
    horizon_squared_errors = []

    for sample_index in range(
        TEST_WINDOWS
    ):

        error = (
            actual_values[sample_index][horizon_index]
            -
            predicted_values[sample_index][horizon_index]
        )

        horizon_absolute_errors.append(
            abs(error)
        )

        horizon_squared_errors.append(
            error ** 2
        )

    horizon_mae = (
        sum(horizon_absolute_errors)
        /
        len(horizon_absolute_errors)
    )

    horizon_rmse = math.sqrt(
        sum(horizon_squared_errors)
        /
        len(horizon_squared_errors)
    )

    horizon_metrics.append({
        "horizon_day": horizon_index + 1,
        "mae": horizon_mae,
        "rmse": horizon_rmse
    })

    print(
        f"Day +{horizon_index + 1}: "
        f"MAE={horizon_mae:.4f}, "
        f"RMSE={horizon_rmse:.4f}"
    )


# ============================================================
# 18. SAVE TEST FORECASTS
# ============================================================

with open(
    FORECAST_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "window",
        "horizon_day",
        "forecast_date",
        "actual_sales",
        "predicted_sales",
        "absolute_error"
    ])

    for sample_index in range(
        TEST_WINDOWS
    ):

        for horizon_index in range(
            FORECAST_HORIZON
        ):

            actual_sale = actual_values[
                sample_index
            ][horizon_index]

            predicted_sale = predicted_values[
                sample_index
            ][horizon_index]

            writer.writerow([
                sample_index + 1,
                horizon_index + 1,
                test_target_dates[
                    sample_index
                ][horizon_index].isoformat(),
                round(actual_sale, 4),
                round(predicted_sale, 4),
                round(
                    abs(
                        actual_sale
                        - predicted_sale
                    ),
                    4
                )
            ])


# ============================================================
# 19. SAVE METRICS
# ============================================================

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "store",
        "item",
        "lookback",
        "forecast_horizon",
        "overall_mae",
        "overall_rmse"
    ])

    writer.writerow([
        STORE_ID,
        ITEM_ID,
        LOOKBACK,
        FORECAST_HORIZON,
        round(overall_mae, 4),
        round(overall_rmse, 4)
    ])


with open(
    HORIZON_METRICS_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "horizon_day",
        "mae",
        "rmse"
    ])

    for result in horizon_metrics:

        writer.writerow([
            result["horizon_day"],
            round(result["mae"], 4),
            round(result["rmse"], 4)
        ])


# ============================================================
# 20. SAVE TRAINING HISTORY
# ============================================================

history_keys = list(
    history.history.keys()
)

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["epoch"] + history_keys
    )

    epoch_count = len(
        history.history[
            history_keys[0]
        ]
    )

    for epoch_index in range(
        epoch_count
    ):

        row = [epoch_index + 1]

        for key in history_keys:
            row.append(
                history.history[key][
                    epoch_index
                ]
            )

        writer.writerow(row)


# ============================================================
# 21. ATTENTION WEIGHTS FOR LATEST TEST WINDOW
# ============================================================

# attention_model shares the same trained layer objects
# used by forecast_model during this execution.
latest_test_window = X_test[-1:]

latest_attention = attention_model.predict(
    latest_test_window,
    verbose=0
)[0]

# Expected:
# forecast horizon x historical look-back
print(
    "\nAttention shape:",
    latest_attention.shape
)


with open(
    ATTENTION_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["forecast_step"]
        +
        [
            f"history_day_{day}"
            for day in range(
                -LOOKBACK,
                0
            )
        ]
    )

    for future_index in range(
        FORECAST_HORIZON
    ):

        writer.writerow(
            [f"day_+{future_index + 1}"]
            +
            [
                round(
                    float(weight),
                    6
                )
                for weight
                in latest_attention[
                    future_index
                ]
            ]
        )


# ============================================================
# 22. CREATE ATTENTION HEATMAP USING TENSORFLOW
# ============================================================

attention_tensor = tf.convert_to_tensor(
    latest_attention,
    dtype=tf.float32
)

minimum_attention = tf.reduce_min(
    attention_tensor
)

maximum_attention = tf.reduce_max(
    attention_tensor
)

attention_range = (
    maximum_attention
    - minimum_attention
)

normalized_attention = tf.where(
    attention_range > 0,
    (
        attention_tensor
        - minimum_attention
    ) / attention_range,
    tf.zeros_like(attention_tensor)
)

# Create an RGB heatmap:
# low value = dark blue
# high value = yellow/red
red = normalized_attention

green = tf.sqrt(
    normalized_attention
)

blue = (
    1.0
    - normalized_attention
)

heatmap_rgb = tf.stack(
    [
        red,
        green,
        blue
    ],
    axis=-1
)

# Increase image size for readability.
heatmap_rgb = tf.image.resize(
    heatmap_rgb,
    size=(
        FORECAST_HORIZON * 50,
        LOOKBACK * 20
    ),
    method="nearest"
)

heatmap_uint8 = tf.cast(
    heatmap_rgb * 255,
    tf.uint8
)

encoded_heatmap = tf.io.encode_png(
    heatmap_uint8
)

tf.io.write_file(
    ATTENTION_IMAGE_PATH,
    encoded_heatmap
)


# ============================================================
# 23. SAVE CONFIGURATION
# ============================================================

configuration = {
    "store_id": STORE_ID,
    "item_id": ITEM_ID,
    "lookback": LOOKBACK,
    "forecast_horizon": FORECAST_HORIZON,
    "sales_minimum": sales_minimum,
    "sales_maximum": sales_maximum,
    "sales_range": sales_range,
    "data_path": DATA_PATH
}

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        configuration,
        file,
        indent=4
    )


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("Model             :", MODEL_PATH)
print("Configuration     :", CONFIG_PATH)
print("Forecasts         :", FORECAST_PATH)
print("Overall metrics   :", METRICS_PATH)
print("Horizon metrics   :", HORIZON_METRICS_PATH)
print("Training history  :", HISTORY_PATH)
print("Attention weights :", ATTENTION_PATH)
print("Attention heatmap :", ATTENTION_IMAGE_PATH)