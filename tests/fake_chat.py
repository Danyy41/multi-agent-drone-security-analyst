"""LangChain fake chat model that supports with_structured_output, for offline tests."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import PrivateAttr


class FakeToolCallingChatModel(GenericFakeChatModel):
    """GenericFakeChatModel plus bind_tools, so LangChain's real structured-output
    parsing (PydanticToolsParser) runs on the scripted AIMessages."""

    _prompts: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tools: list[Any] = PrivateAttr(default_factory=list)

    @property
    def prompts(self) -> list[list[BaseMessage]]:
        return self._prompts

    @property
    def bound_tools(self) -> list[Any]:
        return self._bound_tools

    def bind_tools(self, tools, **kwargs):
        self._bound_tools.extend(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._prompts.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def narrative_message(**args: Any) -> AIMessage:
    """An AIMessage carrying a CommanderNarrative tool call with the given arguments."""
    return AIMessage(content="", tool_calls=[{"name": "CommanderNarrative", "args": args, "id": "call_1"}])


def fake_model(*messages: AIMessage) -> FakeToolCallingChatModel:
    return FakeToolCallingChatModel(messages=iter(messages))
