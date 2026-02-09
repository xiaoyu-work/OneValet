"""
01_greeting.py - Simplest FlowAgents example
"""

import asyncio
from flowagents import flowagent, StandardAgent, InputField, AgentStatus, Orchestrator, OpenAIClient


@flowagent(triggers=["greet", "hello", "hi"])
class GreetingAgent(StandardAgent):
    name = InputField("What's your name?")

    async def on_running(self, msg):
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Hello, {self.name}!"
        )


async def main():
    llm = OpenAIClient(api_key="sk-xxx", model="gpt-4o-mini")
    orchestrator = Orchestrator(llm_client=llm)
    await orchestrator.initialize()

    print("User: Hi")
    result = await orchestrator.handle_message("user_1", "Hi")
    print(f"Agent: {result.raw_message}")

    print("\nUser: Alice")
    result = await orchestrator.handle_message("user_1", "Alice")
    print(f"Agent: {result.raw_message}")


if __name__ == "__main__":
    asyncio.run(main())
