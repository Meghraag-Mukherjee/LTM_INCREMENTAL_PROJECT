import os

import pandas as pd

from src.llm_client import call_llm


DATA_PATH = (
    "data/amazon_reviews/reviews.csv"
)

OUTPUT_PATH = (
    "results/review_summaries.csv"
)


# ============================================================
# 1. REVIEW SUMMARY PROMPT
# ============================================================

SUMMARY_SYSTEM_PROMPT = """
You are RetailMind's product review analyst.

Your job is to summarize customer feedback accurately.

Rules:

1. Use only information contained in the provided reviews.
2. Do not invent product problems or benefits.
3. Identify recurring themes.
4. Separate positive and negative themes.
5. Mention important product issues.
6. Keep the summary concise.
7. Do not mention individual reviewers.

Return the response with these sections:

Overall Summary:
Positive Themes:
Negative Themes:
Product Issues:
Customer Suggestions:
"""


# ============================================================
# 2. SUMMARIZATION FUNCTION
# ============================================================

def summarize_reviews(
    reviews
):

    clean_reviews = [
        str(review).strip()

        for review in reviews

        if str(review).strip()
    ]


    if not clean_reviews:

        return (
            "No reviews available."
        )


    joined_reviews = "\n\n".join(

        f"Review {index + 1}: {review}"

        for index, review
        in enumerate(clean_reviews)
    )


    user_prompt = f"""
Summarize the following customer reviews.

REVIEWS:

{joined_reviews}
"""


    messages = [

        {
            "role": "system",
            "content": SUMMARY_SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": user_prompt
        }
    ]


    return call_llm(
        messages,
        temperature=0.2
    )


# ============================================================
# 3. DATASET SUMMARIZATION
# ============================================================

def summarize_dataset():

    if not os.path.exists(
        DATA_PATH
    ):

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATA_PATH}"
        )


    reviews_df = pd.read_csv(
        DATA_PATH
    )


    if "review" not in reviews_df.columns:

        raise ValueError(
            "reviews.csv must contain "
            "a 'review' column."
        )


    # If product_id exists, generate one
    # summary for each product.
    if (
        "product_id"
        in reviews_df.columns
    ):

        grouped_reviews = (
            reviews_df.groupby(
                "product_id"
            )
        )

    else:

        # Otherwise treat the dataset
        # as one review collection.
        grouped_reviews = [
            (
                "all_products",
                reviews_df
            )
        ]


    summary_rows = []


    for (
        product_id,
        product_reviews
    ) in grouped_reviews:

        reviews = (
            product_reviews[
                "review"
            ]
            .dropna()
            .tolist()
        )


        # Limit each direct request to a
        # manageable review batch.
        reviews = reviews[:50]


        print(
            "\nSummarizing:",
            product_id
        )


        summary = summarize_reviews(
            reviews
        )


        summary_rows.append({
            "product_id": product_id,

            "review_count": (
                len(reviews)
            ),

            "summary": summary
        })


    os.makedirs(
        "results",
        exist_ok=True
    )


    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        "\nSummaries saved:",
        OUTPUT_PATH
    )


    return summary_df


# ============================================================
# 4. RUN
# ============================================================

if __name__ == "__main__":

    summaries = (
        summarize_dataset()
    )


    print(
        "\nGenerated summaries:"
    )


    print(
        summaries.to_string(
            index=False
        )
    )