import pytest
from langsmith.wrappers.openai import openai


def test_openai_chat_completion():
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in san francisco right now?",
            }
        ],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
    )
    assert result.choices[0].message.function_call.name == "get_current_weather"


@pytest.mark.asyncio
async def test_openai_chat_completion_async():
    result = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"What's the weather like in San Francisco right now?",
            }
        ],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
    )
    assert result.choices[0].message.function_call.name == "get_current_weather"


def test_openai_completion():
    result = openai.Completion.create(
        model="text-davinci-003",
        prompt="Say 'foo'",
    )
    assert "foo" in result.choices[0].text.lower()


@pytest.mark.asyncio
async def test_openai_completion_async():
    result = await openai.Completion.acreate(
        model="text-davinci-003",
        prompt="Say 'foo'",
    )
    assert "foo" in result.choices[0].text.lower()
