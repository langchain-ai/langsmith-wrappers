from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langsmith.wrappers.base import ModuleWrapper

if TYPE_CHECKING:
    import openai


def __getattr__(name: str) -> Any:
    if name == "openai":
        try:
            import openai as openai_base
        except ImportError:
            raise ImportError(
                "OpenAI SDK is not installed. "
                "Please install it with `pip install openai`."
            )

        return ModuleWrapper(
            openai_base,
            llm_paths={
                "openai.api_resources.chat_completion.create",
                "openai.api_resources.chat_completion.acreate",
                "openai.api_resources.completion.create",
                "openai.api_resources.completion.acreate",
            },
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "openai",  # noqa: F822
]
