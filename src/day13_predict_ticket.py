import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_DIRECTORY = (
    "models/bert_ticket_router"
)


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


def route_ticket(ticket_text):

    encoded_ticket = tokenizer(
        ticket_text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=256
    )


    encoded_ticket = {
        key: value.to(device)

        for key, value
        in encoded_ticket.items()
    }


    with torch.no_grad():

        output = model(
            **encoded_ticket
        )


    probabilities = torch.softmax(
        output.logits,
        dim=-1
    )[0]


    predicted_id = int(
        torch.argmax(
            probabilities
        ).item()
    )


    predicted_category = (
        model.config.id2label[
            predicted_id
        ]
    )


    all_probabilities = {
        model.config.id2labelround(
            float(
                probability.item()
            ),
            4
        )

        for index, probability
        in enumerate(probabilities)
    }


    return {
        "ticket": ticket_text,

        "category": predicted_category,

        "confidence": round(
            float(
                probabilities[
                    predicted_id
                ].item()
            ),
            4
        ),

        "category_probabilities": (
            all_probabilities
        )
    }


if __name__ == "__main__":

    sample_tickets = [
        "My payment was deducted twice.",

        (
            "The item arrived damaged "
            "and does not work."
        ),

        (
            "I forgot my password and "
            "cannot access my account."
        ),

        (
            "My package has not arrived "
            "after ten days."
        )
    ]


    for ticket in sample_tickets:

        result = route_ticket(
            ticket
        )

        print("\n" + "-" * 70)

        print(
            "Ticket:",
            result["ticket"]
        )

        print(
            "Route:",
            result["category"]
        )

        print(
            "Confidence:",
            result["confidence"]
        )