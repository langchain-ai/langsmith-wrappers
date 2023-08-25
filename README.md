# LangSmith API Wrappers


Some functionality to wrap common apis (e.g., `openai`) with [LangSmith](https://smith.langchain.com/) instrumentation.


Example:

```python
from langsmith.wrappers.openai import openai

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
print(result)
```


You can try wrapping your own library. Function calls will be mapped as "chain" runs.

Example:

```python
from langsmith.wrappers.base import ModuleWrapper
import transformers as transformers_base
transformers = ModuleWrapper(transformers_base)

pipe = transformers.pipeline("text2text-generation", model="google/t5-efficient-tiny")
result = pipe("This is a test")
```