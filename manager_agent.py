import asyncio
import os
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# Import sub-agents
from database_agent import agent as database_agent
from summary_agent import summary_agent
from feature_agent import feature_agent

load_dotenv()

# --------- Agent + Provider Setup ---------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

manager_agent = Agent(
    model=model,
    system_prompt=(
        "You are the Manager Agent for the Meeting System.\n\n"
        "Your workflow is strictly sequential:\n"
        "1. Always save the meeting first.\n"
        "2. After saving, generate meeting summary.\n"
        "3. After the summary, extract ONLY the meeting features.\n\n"
        "Rules:\n"
        "- Do not skip or change the order: Save → Summarize → Extract Features.\n"
        "- Confirm with the user before performing saving.\n"
        "- Keep responses short and clear."
    ),
)



# --- Manager Tools ---
@manager_agent.tool
async def save_or_update_meeting(ctx: RunContext[None], file_path: str) -> str:
    """Save or update meeting via database_agent"""
    
    print("\n------- Manager Save_or_update_meeting Tool -------\n")
    
    print(f"\n[DEBUG] Manager: Saving meeting from {file_path}\n")

    # Step 1 - Save meeting using Database Agent
    db_result = await database_agent.run(
        f'Save the meeting from "{file_path}"'
    )
    print(f"\n[DEBUG] Database Agent Output: {db_result.output}\n")

    return (
        f"meeting saved.\n"
        f"Database Response: {db_result.output}\n"
    )
    
# --- Manager Tools ---
@manager_agent.tool
async def summarize_meeting(ctx: RunContext[None], meeting_name: str, file_path) -> str:
    """Summarize the meeting via summary_agent"""
    
    print("\n------- Manager Summarize meeting Tool -------\n")

    # Step 1 - Generate summary using Summary Agent
    print(f"\n[DEBUG] Manager: Summarizing meeting {meeting_name} from file path: {file_path}\n")
    summary_result = await summary_agent.run(
        f"Summarize the meeting: {meeting_name} from file path: {file_path}"
    )
    print(f"\n[DEBUG] Summary Agent Output: {summary_result.output}\n")

    return (
        f"Summary: {summary_result.output}"
    )

# --- Manager Tools ---
@manager_agent.tool
async def extract_features(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """Extract meeting features via feature_agent"""

    print("\n------- Manager Extract Features Tool -------\n")
    print(f"[DEBUG] Manager: Extracting features from {meeting_name} at {file_path}\n")

    # Call feature agent tool
    result = await feature_agent.run(
        f'Extract features for meeting "{meeting_name}" from file "{file_path}"'
    )

    print(f"[DEBUG] Feature Agent Output: {result.output}\n")
    return f"Features extracted for meeting '{meeting_name}'.\n{result.output}"


# --- Main Interactive Loop ---
async def main():
    print("\n[DEBUG] Manager Agent interactive session started.")
    print("Type 'exit' to quit.\n")

    message_history = []

    # Initial greeting
    response = await manager_agent.run("Hi!", message_history=message_history)
    print(f"{response.output}\n")
    message_history.extend(response.new_messages())

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("\n[DEBUG] Session ended, Goodbye!\n")
            break

        response = await manager_agent.run(user_input, message_history=message_history)
        print(f"\n[meeting Assistant Response]\n{response.output}\n")
        message_history.extend(response.new_messages())


if __name__ == "__main__":
    asyncio.run(main())
