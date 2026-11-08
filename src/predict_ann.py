import joblib
import pandas as pd

from tensorflow import keras


ANN_MODEL_PATH = "models/churn_ann.keras"

PREPROCESSOR_PATH = (
    "models/ann_preprocessor.joblib"
)


# ============================================================
# 1. LOAD MODEL AND PREPROCESSOR
# ============================================================

ann_model = keras.models.load_model(
    ANN_MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


# ============================================================
# 2. PREDICTION FUNCTION
# ============================================================

def predict_churn(
    customer_data,
    threshold=0.50
):
    """
    Predict customer churn using the Keras ANN.

    Parameters
    ----------
    customer_data : dict
        Customer information.

    threshold : float
        Probability threshold used to convert the
        ANN probability into class 0 or 1.

    Returns
    -------
    dict
        Predicted class, label and probability.
    """

    customer_df = pd.DataFrame(
        [customer_data]
    )

    if "TotalCharges" in customer_df.columns:
        customer_df["TotalCharges"] = (
            pd.to_numeric(
                customer_df["TotalCharges"],
                errors="coerce"
            )
        )

    processed_data = preprocessor.transform(
        customer_df
    )

    processed_data = processed_data.astype(
        "float32"
    )

    churn_probability = float(
        ann_model.predict(
            processed_data,
            verbose=0
        )[0][0]
    )

    churn_prediction = int(
        churn_probability >= threshold
    )

    return {
        "churn_prediction": churn_prediction,

        "churn_label": (
            "Yes"
            if churn_prediction == 1
            else "No"
        ),

        "churn_probability": round(
            churn_probability,
            4
        ),

        "threshold": threshold
    }


# ============================================================
# 3. TEST CUSTOMER
# ============================================================

if __name__ == "__main__":

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.50,
        "TotalCharges": 966.00
    }

    result = predict_churn(
        sample_customer
    )

    print("\nANN churn prediction:")

    print(result)