"""
FlowAgent Message System - Core message types for multi-modal content

Supports text, image, audio, video, and tool call content blocks.
"""

from dataclasses import dataclass, field
from typing import Literal, List, Union, Optional
from datetime import datetime
from uuid import uuid4


# ===== Content Blocks =====

@dataclass
class TextBlock:
    """Text content block"""
    text: str
    type: Literal["text"] = "text"


@dataclass
class ImageBlock:
    """Image content block"""
    source: dict  # {"type": "base64", "data": "..."} or {"type": "url", "url": "..."}
    type: Literal["image"] = "image"


@dataclass
class AudioBlock:
    """Audio content block"""
    source: dict  # {"type": "base64", "data": "..."} or {"type": "url", "url": "..."}
    type: Literal["audio"] = "audio"


@dataclass
class VideoBlock:
    """Video content block"""
    source: dict
    type: Literal["video"] = "video"


@dataclass
class ToolUseBlock:
    """Tool call request block"""
    name: str
    input: dict
    id: str = ""
    type: Literal["tool_use"] = "tool_use"

    def __post_init__(self):
        if not self.id:
            self.id = f"tool_{uuid4().hex[:8]}"


@dataclass
class ToolResultBlock:
    """Tool call result block"""
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


# Union type for all content blocks
ContentBlock = Union[TextBlock, ImageBlock, AudioBlock, VideoBlock, ToolUseBlock, ToolResultBlock]


# ===== Message =====

@dataclass
class Message:
    """
    Core message class for FlowAgent framework

    Supports both simple text and multimodal content.

    Examples:
        # Simple text message
        msg = Message(name="assistant", content="Hello!", role="assistant")

        # Multimodal message
        msg = Message(
            name="user",
            content=[
                TextBlock(text="What's in this image?"),
                ImageBlock(source={"type": "url", "url": "https://..."})
            ],
            role="user"
        )
    """
    name: str
    content: Union[str, List[ContentBlock]]
    role: Literal["user", "assistant", "system"]
    metadata: Optional[dict] = None
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def get_text(self) -> str:
        """
        Get pure text content from message

        Returns:
            Combined text from all TextBlocks, or the string content
        """
        if isinstance(self.content, str):
            return self.content

        texts = []
        for block in self.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))

        return "".join(texts)

    def get_blocks(self, block_type: Optional[str] = None) -> List[ContentBlock]:
        """
        Get content blocks, optionally filtered by type

        Args:
            block_type: Filter by type ("text", "image", "audio", "tool_use", "tool_result")

        Returns:
            List of content blocks
        """
        if isinstance(self.content, str):
            blocks = [TextBlock(text=self.content)]
        else:
            blocks = self.content or []

        if block_type:
            return [b for b in blocks if getattr(b, 'type', None) == block_type or
                    (isinstance(b, dict) and b.get('type') == block_type)]

        return blocks

    def has_blocks(self, block_type: str) -> bool:
        """Check if message has blocks of given type"""
        return len(self.get_blocks(block_type)) > 0

    def to_dict(self) -> dict:
        """Convert message to dictionary"""
        content = self.content
        if isinstance(self.content, list):
            content = []
            for block in self.content:
                if hasattr(block, '__dict__'):
                    content.append({k: v for k, v in block.__dict__.items()})
                else:
                    content.append(block)

        return {
            "id": self.id,
            "name": self.name,
            "content": content,
            "role": self.role,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create message from dictionary"""
        return cls(
            name=data["name"],
            content=data["content"],
            role=data["role"],
            metadata=data.get("metadata"),
            id=data.get("id", uuid4().hex[:12]),
            timestamp=data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

    def __repr__(self) -> str:
        content_preview = self.get_text()[:50] + "..." if len(self.get_text()) > 50 else self.get_text()
        return f"Message(name='{self.name}', role='{self.role}', content='{content_preview}')"
