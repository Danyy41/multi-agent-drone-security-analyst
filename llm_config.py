"""OpenAI settings shared by the CrewAI crew and the LangChain report chain."""

import os

DEFAULT_MODEL = "gpt-5"


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM is needed but no OpenAI API key is configured."""


def openai_settings() -> tuple[str, str]:
    """Return (api_key, model) from the environment, failing clearly when no key is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Export it (or put it in an untracked .env file) "
            "to run the LLM agents, or use the rule-based mode, which needs no key."
        )
    return api_key, os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
