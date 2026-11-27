import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_DIRECTORY = (
    "models/bert_sentiment"
)


# ============================================================
# LOAD MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIRECTORY
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_DIRECTORY
    )
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model.to(device)

model.eval()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sentiment(review):

    encoded_review = tokenizer(
        review,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=256
    )


    encoded_review = {
        key: value.to(device)

        for key, value
        in encoded_review.items()
    }


    with torch.no_grad():

        model_output = model(
            **encoded_review
        )


    probabilities = torch.softmax(
        model_output.logits,
        dim=-1
    )[0]


    predicted_index = int(
        torch.argmax(
            probabilities
        ).item()
    )


    sentiment = model.config.id2label[
        predicted_index
    ]


    return {
        "review": review,

        "sentiment": sentiment,

        "confidence": round(
            float(
                probabilities[
                    predicted_index
                ].item()
            ),
            4
        ),

        "negative_probability": round(
            float(
                probabilities[0].item()
            ),
            4
        ),

        "positive_probability": round(
            float(
                probabilities[1].item()
            ),
            4
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_reviews = [
        (
            "This product is excellent and "
            "works perfectly."
        ),

        (
            "The item broke after two days "
            "and customer support was useless."
        ),

        (
            "The quality is acceptable but "
            "the delivery was very slow."
        )
    ]


    for review in sample_reviews:

        result = predict_sentiment(
            review
        )

        print("\n" + "-" * 70)

        print(
            "Review:",
            result["review"]
        )

        print(
            "Sentiment:",
            result["sentiment"]
        )

        print(
            "Confidence:",
            result["confidence"]
        )