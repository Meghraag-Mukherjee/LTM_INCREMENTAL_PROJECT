import pandas as pd


def add_engineered_features(X):
    """
    Create RetailMind churn features.

    This function does not modify the original DataFrame.
    """

    X = X.copy()

    # ------------------------------------------------
    # Ensure numerical columns are numeric
    # ------------------------------------------------

    numeric_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for column in numeric_columns:
        if column in X.columns:
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    # ------------------------------------------------
    # 1. Tenure buckets
    # ------------------------------------------------

    if "tenure" in X.columns:
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[
                -1,
                12,
                24,
                48,
                60,
                float("inf")
            ],
            labels=[
                "0-12",
                "13-24",
                "25-48",
                "49-60",
                "61+"
            ]
        ).astype("object")

    # ------------------------------------------------
    # 2. Average charge per month
    # ------------------------------------------------

    if {
        "TotalCharges",
        "tenure"
    }.issubset(X.columns):

        tenure_safe = X["tenure"].replace(
            0,
            1
        )

        X["average_charge_per_month"] = (
            X["TotalCharges"] /
            tenure_safe
        )

    # ------------------------------------------------
    # 3. Charge difference
    # ------------------------------------------------

    if {
        "TotalCharges",
        "MonthlyCharges",
        "tenure"
    }.issubset(X.columns):

        expected_total = (
            X["MonthlyCharges"] *
            X["tenure"]
        )

        X["charge_difference"] = (
            X["TotalCharges"] -
            expected_total
        )

    # ------------------------------------------------
    # 4. Contract risk flag
    # ------------------------------------------------

    if "Contract" in X.columns:
        X["month_to_month_flag"] = (
            X["Contract"] == "Month-to-month"
        ).astype(int)

    # ------------------------------------------------
    # 5. Electronic check flag
    # ------------------------------------------------

    if "PaymentMethod" in X.columns:
        X["electronic_check_flag"] = (
            X["PaymentMethod"] ==
            "Electronic check"
        ).astype(int)

    # ------------------------------------------------
    # 6. Fibre optic flag
    # ------------------------------------------------

    if "InternetService" in X.columns:
        X["fiber_optic_flag"] = (
            X["InternetService"] ==
            "Fiber optic"
        ).astype(int)

    # ------------------------------------------------
    # 7. Automatic payment flag
    # ------------------------------------------------

    if "PaymentMethod" in X.columns:
        X["automatic_payment_flag"] = (
            X["PaymentMethod"].isin([
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
        ).astype(int)

    # ------------------------------------------------
    # 8. Count optional services
    # ------------------------------------------------

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    available_services = [
        column
        for column in service_columns
        if column in X.columns
    ]

    if available_services:
        X["number_of_services"] = (
            X[available_services]
            .eq("Yes")
            .sum(axis=1)
        )

    # ------------------------------------------------
    # 9. Security/support flag
    # ------------------------------------------------

    support_columns = [
        column
        for column in [
            "OnlineSecurity",
            "TechSupport"
        ]
        if column in X.columns
    ]

    if support_columns:
        X["has_security_or_support"] = (
            X[support_columns]
            .eq("Yes")
            .any(axis=1)
            .astype(int)
        )

    # ------------------------------------------------
    # 10. Senior customer with no partner
    # ------------------------------------------------

    if {
        "SeniorCitizen",
        "Partner"
    }.issubset(X.columns):

        X["senior_without_partner"] = (
            (X["SeniorCitizen"] == 1) &
            (X["Partner"] == "No")
        ).astype(int)

    return X