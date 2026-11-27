from src.review_summarizer import (
    summarize_reviews
)

from src.support_chatbot import (
    RetailMindChatbot
)


class RetailMindAssistant:
    """
    Main Day-14 assistant module.

    Capabilities:

    1. Product review summarization
    2. Customer-support chatbot
    """


    def __init__(self):

        self.chatbot = (
            RetailMindChatbot()
        )


    def summarize(
        self,
        reviews
    ):

        return summarize_reviews(
            reviews
        )


    def chat(
        self,
        customer_message
    ):

        return self.chatbot.ask(
            customer_message
        )


    def reset_chat(
        self
    ):

        self.chatbot.reset()