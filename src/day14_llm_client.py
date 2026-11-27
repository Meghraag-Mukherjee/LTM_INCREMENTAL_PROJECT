import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


API_KEY = os.getenv(
    "LLM_API_KEY"
)

MODEL_NAME = os.getenv(
    "LLM_MODEL"
)

BASE_URL = os.getenv(
    "LLM_BASE_URL"
)


# ============================================================
# 2. VALIDATE SETTINGS
# ============================================================

if not API_KEY:
    raise ValueError(
        "LLM_API_KEY is missing from .env"
    )


if not MODEL_NAME:
    raise ValueError(
        "LLM_MODEL is missing from .env"
    )


# ============================================================
# 3. CREATE CLIENT
# ============================================================

if BASE_URL:

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

else:

    client = OpenAI(
        api_key=API_KEY
    )


# ============================================================
# 4. THIN LLM CLIENT
# ============================================================

def call_llm(
    messages,
    temperature=0.2
):
    """
    Send messages to the configured LLM.
    """

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=messages,

        temperature=temperature
    )


    return (
        response
        .choices[0]
        .message
        .content
    )