import joblib
import pandas as pd

# Import required for loading the persisted pipeline.
from src.features import add_engineered_features


MODEL_PATH = "models/churn_final.joblib"

model = joblib.load(MODEL_PATH)


def predict(customer_data):
    """
    Predict whether a customer will churn.

    Parameters
    ----------
    customer_data : dict
        One customer's feature values.

    Returns
    -------
    dict
        Prediction, label, and churn probability.
    """

    customer_df = pd.DataFrame(
        [customer_data]
    )

    prediction = int(
        model.predict(customer_df)[0]
    )

    churn_probability = float(
        model.predict_proba(
            customer_df
        )[0][1]
    )

    return {
        "churn_prediction": prediction,

        "churn_label": (
            "Yes"
            if prediction == 1
            else "No"
        ),

        "churn_probability": round(
            churn_probability,
            4
        )
    }


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

    result = predict(
        sample_customer
    )

    print("\nFinal churn prediction:")
    print(result)