import re

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
nltk.download(
    "stopwords",
    quiet=True
)

nltk.download(
    "wordnet",
    quiet=True
)


STOP_WORDS = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean one product review.

    Steps:
    1. lowercase
    2. remove URLs
    3. remove HTML
    4. remove punctuation/numbers
    5. tokenize
    6. remove stopwords
    7. lemmatize
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep letters only
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Simple tokenization
    tokens = text.split()

    cleaned_tokens = []

    for token in tokens:

        if (
            token not in STOP_WORDS
            and len(token) > 1
        ):

            token = lemmatizer.lemmatize(
                token
            )

            cleaned_tokens.append(
                token
            )

    return " ".join(
        cleaned_tokens
    )