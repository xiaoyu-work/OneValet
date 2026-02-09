"""
02_restaurant_booking.py - Field collection with validation
"""

import asyncio
from flowagents import flowagent, StandardAgent, InputField, AgentStatus, Orchestrator, OpenAIClient


def validate_guests(value: str) -> bool:
    if not value.isdigit():
        raise ValueError("Please enter a number")
    num = int(value)
    if num < 1 or num > 20:
        raise ValueError("We can accommodate 1-20 guests")
    return True


@flowagent(triggers=["book table", "reserve", "reservation", "book restaurant"])
class RestaurantBookingAgent(StandardAgent):
    guests = InputField("How many guests?", validator=validate_guests)
    date = InputField("What date?")
    name = InputField("Name for the reservation?")

    async def on_running(self, msg):
        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f"Booked for {self.guests} on {self.date} under {self.name}!"
        )


async def main():
    llm = OpenAIClient(api_key="sk-xxx", model="gpt-4o-mini")
    orchestrator = Orchestrator(llm_client=llm)
    await orchestrator.initialize()

    conversations = [
        ("I'd like to book a table", "How many guests?"),
        ("4", "What date?"),
        ("Friday evening", "Name for the reservation?"),
        ("John Smith", "Booked for"),
    ]

    for user_input, expected in conversations:
        print(f"User: {user_input}")
        result = await orchestrator.handle_message("user_1", user_input)
        print(f"Agent: {result.raw_message}\n")


if __name__ == "__main__":
    asyncio.run(main())
