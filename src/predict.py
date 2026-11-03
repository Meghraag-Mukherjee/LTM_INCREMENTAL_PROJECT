import pandas as pd
import joblib


MODEL_PATH = "models/churn_v3.joblib"


# Load trained pipeline
model = joblib.load(MODEL_PATH)


def predict(customer_data):
    """
    Predict customer churn.

    Parameters
    ----------
    customer_data : dict
        Customer information.

    Returns
    -------
    dict
        Predicted class and churn probability.
    """

    df = pd.DataFrame([customer_data])

    # Match training cleaning
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }


if __name__ == "__main__":

    customer = {
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

    result = predict(customer)

    print(result)