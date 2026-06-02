"""消息系统"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """消息类"""

    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（OpenAI API 格式）"""
        return {
            "role": self.role,
            "content": self.content,
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
