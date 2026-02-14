"""
OneValet Base Agent - Foundation class for all agents

Provides:
- Simple hook system (pre/post reply)
- Unique ID generation
- Basic async interface
"""

import logging
from typing import Callable, Optional, Dict
from uuid import uuid4
from collections import OrderedDict

from .message import Message

logger = logging.getLogger(__name__)


class BaseAgent:
    """
    Base class for all OneValet agents

    Features:
    - Unique agent ID
    - Hook system for pre/post processing
    - Async reply interface

    Example:
        class MyAgent(BaseAgent):
            async def _do_reply(self, msg: Message) -> Message:
                return Message(
                    name="MyAgent",
                    content="Hello!",
                    role="assistant"
                )

        agent = MyAgent(name="my-agent")
        response = await agent.reply(Message(...))
    """

    # Class-level hooks (apply to all instances of this specific class)
    # Each subclass gets its own copy via __init_subclass__.
    _class_pre_reply_hooks: Dict[str, Callable] = OrderedDict()
    _class_post_reply_hooks: Dict[str, Callable] = OrderedDict()

    def __init_subclass__(cls, **kwargs):
        """Give each subclass its own hook dicts so they don't share state."""
        super().__init_subclass__(**kwargs)
        cls._class_pre_reply_hooks = OrderedDict(cls._class_pre_reply_hooks)
        cls._class_post_reply_hooks = OrderedDict(cls._class_post_reply_hooks)

    def __init__(self, name: Optional[str] = None, **kwargs):
        """
        Initialize base agent

        Args:
            name: Agent name (defaults to class name)
        """
        self.id = uuid4().hex[:8]
        self.name = name or self.__class__.__name__

        # Instance-level hooks
        self._instance_pre_reply_hooks: Dict[str, Callable] = OrderedDict()
        self._instance_post_reply_hooks: Dict[str, Callable] = OrderedDict()

    async def reply(self, msg: Message = None) -> Message:
        """
        Main entry point - handles hooks and calls _do_reply

        Args:
            msg: Input message

        Returns:
            Response message
        """
        # Run pre-reply hooks
        msg = await self._run_pre_hooks(msg)

        # Call actual implementation
        result = await self._do_reply(msg)

        # Run post-reply hooks
        result = await self._run_post_hooks(msg, result)

        return result

    async def _do_reply(self, msg: Message) -> Message:
        """
        Actual reply implementation - override in subclasses

        Args:
            msg: Input message

        Returns:
            Response message
        """
        raise NotImplementedError(
            f"_do_reply not implemented in {self.__class__.__name__}"
        )

    async def _run_pre_hooks(self, msg: Message) -> Message:
        """Run all pre-reply hooks"""
        # Class-level hooks first
        for name, hook in self._class_pre_reply_hooks.items():
            try:
                result = hook(self, msg)
                if result is not None:
                    msg = result
            except Exception as e:
                logger.error(f"Pre-reply hook '{name}' failed: {e}")

        # Instance-level hooks
        for name, hook in self._instance_pre_reply_hooks.items():
            try:
                result = hook(self, msg)
                if result is not None:
                    msg = result
            except Exception as e:
                logger.error(f"Pre-reply hook '{name}' failed: {e}")

        return msg

    async def _run_post_hooks(self, input_msg: Message, output_msg: Message) -> Message:
        """Run all post-reply hooks"""
        # Class-level hooks first
        for name, hook in self._class_post_reply_hooks.items():
            try:
                result = hook(self, input_msg, output_msg)
                if result is not None:
                    output_msg = result
            except Exception as e:
                logger.error(f"Post-reply hook '{name}' failed: {e}")

        # Instance-level hooks
        for name, hook in self._instance_post_reply_hooks.items():
            try:
                result = hook(self, input_msg, output_msg)
                if result is not None:
                    output_msg = result
            except Exception as e:
                logger.error(f"Post-reply hook '{name}' failed: {e}")

        return output_msg

    # ===== Hook Management =====

    @classmethod
    def register_class_hook(
        cls,
        hook_type: str,
        hook_name: str,
        hook: Callable
    ) -> None:
        """
        Register a class-level hook (applies to all instances)

        Args:
            hook_type: "pre_reply" or "post_reply"
            hook_name: Unique name for the hook
            hook: Hook function
                - pre_reply: (agent, msg) -> msg or None
                - post_reply: (agent, input_msg, output_msg) -> msg or None
        """
        if hook_type == "pre_reply":
            cls._class_pre_reply_hooks[hook_name] = hook
        elif hook_type == "post_reply":
            cls._class_post_reply_hooks[hook_name] = hook
        else:
            raise ValueError(f"Invalid hook_type: {hook_type}")

    @classmethod
    def remove_class_hook(cls, hook_type: str, hook_name: str) -> None:
        """Remove a class-level hook"""
        if hook_type == "pre_reply":
            cls._class_pre_reply_hooks.pop(hook_name, None)
        elif hook_type == "post_reply":
            cls._class_post_reply_hooks.pop(hook_name, None)

    @classmethod
    def clear_class_hooks(cls, hook_type: Optional[str] = None) -> None:
        """Clear class-level hooks"""
        if hook_type is None or hook_type == "pre_reply":
            cls._class_pre_reply_hooks.clear()
        if hook_type is None or hook_type == "post_reply":
            cls._class_post_reply_hooks.clear()

    def register_instance_hook(
        self,
        hook_type: str,
        hook_name: str,
        hook: Callable
    ) -> None:
        """
        Register an instance-level hook (applies only to this instance)

        Args:
            hook_type: "pre_reply" or "post_reply"
            hook_name: Unique name for the hook
            hook: Hook function
        """
        if hook_type == "pre_reply":
            self._instance_pre_reply_hooks[hook_name] = hook
        elif hook_type == "post_reply":
            self._instance_post_reply_hooks[hook_name] = hook
        else:
            raise ValueError(f"Invalid hook_type: {hook_type}")

    def remove_instance_hook(self, hook_type: str, hook_name: str) -> None:
        """Remove an instance-level hook"""
        if hook_type == "pre_reply":
            self._instance_pre_reply_hooks.pop(hook_name, None)
        elif hook_type == "post_reply":
            self._instance_post_reply_hooks.pop(hook_name, None)

    def clear_instance_hooks(self, hook_type: Optional[str] = None) -> None:
        """Clear instance-level hooks"""
        if hook_type is None or hook_type == "pre_reply":
            self._instance_pre_reply_hooks.clear()
        if hook_type is None or hook_type == "post_reply":
            self._instance_post_reply_hooks.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.id}', name='{self.name}')"
