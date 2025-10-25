"""
Utility to initialize and return a configured LLM for use in LangChain pipelines.
"""

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


def get_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    api_key: str | None = None,
):
    """
    Initializes and returns an OpenAI Chat model instance.

    Args:
        model_name (str): The model name to use. Defaults to 'gpt-4o-mini'.
        temperature (float): The creativity level of the model. Defaults to 0.0.
        max_tokens (int): Maximum output tokens. Defaults to 1024.
        api_key (str, optional): Custom OpenAI API key. If not provided, reads from env.

    Returns:
        ChatOpenAI: Configured LLM object for use in LangChain.
    """

    # Read key from environment if not passed explicitly
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment."
        )

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )

    return llm