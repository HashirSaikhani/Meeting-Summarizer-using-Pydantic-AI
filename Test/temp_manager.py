import asyncio
import os
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# Import only feature agent
from feature_agent import feature_agent

load_dotenv()

# --------- Agent + Provider Setup ---------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

temp_manager = Agent(
    model=model,
    system_prompt=(
        "You are the Temp Manager Agent for testing purposes.\n\n"
        "Your only responsibility:\n"
        "- Use the feature agent tool to extract meeting features when requested.\n\n"
    ),
)

# --- Temp Manager Tool (only feature extraction) ---
@temp_manager.tool
async def extract_features(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """Extract meeting features via feature_agent"""

    print("\n------- Temp Manager Extract Features Tool -------\n")
    print(f"[DEBUG] Temp Manager: Extracting features from {meeting_name} at {file_path}\n")

    # Call feature agent directly
    result = await feature_agent.run(
        f'Extract features for meeting "{meeting_name}" from file "{file_path}"'
    )

    print(f"[DEBUG] Feature Agent Output: {result.output}\n")
    return f"Features extracted for meeting '{meeting_name}'.\n{result.output}"


# --- Main Interactive Loop ---
async def main():
    print("\n[DEBUG] Temp Manager Agent interactive session started.")
    print("Type 'exit' to quit.\n")

    message_history = []

    # Initial greeting
    response = await temp_manager.run("please extract features for meeeting named TEMP and path is Examples/2.txt", message_history=message_history)
    print(f"{response.output}\n")
    message_history.extend(response.new_messages())

    # while True:
    #     user_input = input("You: ").strip()
    #     if user_input.lower() in {"exit", "quit"}:
    #         print("\n[DEBUG] Session ended, Goodbye!\n")
    #         break

    #     response = await temp_manager.run(user_input, message_history=message_history)
    #     print(f"\n[Temp Manager Response]\n{response.output}\n")
    #     message_history.extend(response.new_messages())


if __name__ == "__main__":
    asyncio.run(main())
