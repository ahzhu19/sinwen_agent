from abc import ABC
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .config import Config
from .llm_types import LLMToolResponse, ToolCall

# 加载 .env 文件中的环境变量
load_dotenv()

LLMMessages = Iterable[ChatCompletionMessageParam] | Sequence[dict[str, Any]]


class BaseLLM(ABC):
    """
    LLM 客户端基类。

    负责初始化兼容 OpenAI 接口的客户端，并提供同步调用与流式调用能力。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        config: Config | None = None,
    ):
        config = config or Config.from_env()
        model_id = model or config.default_model
        api_key = api_key or config.llm_api_key
        base_url = base_url or config.llm_base_url
        timeout_seconds = timeout or config.llm_timeout

        if not model_id or not api_key or not base_url:
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.model = model_id
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    def _cast_messages(self, messages: LLMMessages) -> Iterable[ChatCompletionMessageParam]:
        return cast(Iterable[ChatCompletionMessageParam], messages)

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        """同步调用聊天补全，返回完整文本。"""
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._cast_messages(messages),
                temperature=temperature,
                stream=False,
                **kwargs,
            )
            content = response.choices[0].message.content if response.choices else None
            print("✅ 大语言模型响应成功")
            return content
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

    def stream_invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式调用聊天补全，逐块产出文本。"""
        print(f"🧠 正在调用 {self.model} 模型（流式）...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._cast_messages(messages),
                temperature=temperature,
                stream=True,
                **kwargs,
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Literal["auto", "none", "required"] | dict[str, Any] = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        """调用聊天补全并支持 Function Calling。"""
        print(f"🧠 正在调用 {self.model} 模型（工具模式）...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._cast_messages(messages),
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                stream=False,
                **kwargs,
            )
            choice = response.choices[0] if response.choices else None
            message = choice.message if choice else None
            if message is None:
                return LLMToolResponse(content=None, tool_calls=None)

            tool_calls: list[ToolCall] | None = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments or "{}",
                    )
                    for tc in message.tool_calls
                ]

            usage: dict[str, Any] = {}
            if response.usage is not None:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            print("✅ 大语言模型响应成功")
            return LLMToolResponse(
                content=message.content,
                tool_calls=tool_calls,
                usage=usage,
            )
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return LLMToolResponse(content=None, tool_calls=None)

    def think(
        self,
        messages: LLMMessages,
        temperature: float = 0,
    ) -> str | None:
        """调用大语言模型进行思考（与 invoke 等价，保留兼容）。"""
        return self.invoke(messages, temperature=temperature)
