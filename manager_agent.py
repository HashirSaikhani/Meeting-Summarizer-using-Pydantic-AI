import asyncio
import os
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# Import sub-agents
from database_agent import agent as database_agent
from summary_agent import summary_agent

load_dotenv()

# --------- Agent + Provider Setup ---------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-1.5-flash", provider=provider)

# --- Manager Agent ---
manager_agent = Agent(
    model=model,
    system_prompt=(
        "You are the Manager Agent for the Transcript System.\n"
        "You help users manage their meeting transcripts by either saving new transcripts or generating summaries of existing ones.\n\n"
        "Your two main capabilities are:\n"
        "1. Saving a meeting transcript when the user provides a meeting name and a transcript file path.\n"
        "   - After saving, you also generate a summary for that meeting.\n"
        "2. Summarizing an existing meeting transcript when the user only asks for a summary.\n\n"
        "Guidelines:\n"
        "- If the user wants to save a meeting, ask for the meeting name and file path if they are missing.\n"
        "- If the user only wants a summary, ask for the meeting name if it’s missing.\n"
        "- Keep responses concise, guiding the user clearly to provide the necessary details.\n"
        "- Do not invent meeting names, file paths, or transcript content.\n"
    ),
)


# --- Manager Tools ---
@manager_agent.tool
async def save_meeting(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """Save transcript via database_agent"""
    
    print("\n------- Save Meeting Tool -------\n")
    
    print(f"[DEBUG] Manager: Saving transcript for {meeting_name} from {file_path}")

    # Step 1 - Save transcript using Database Agent
    db_result = await database_agent.run(
        f'Save the transcript from "{file_path}" with the meeting name "{meeting_name}"'
    )
    print(f"\n[DEBUG] Database Agent Output: {db_result.output}")

    return (
        f"Transcript saved.\n"
        f"Database Response: {db_result.output}\n"
    )
    
# --- Manager Tools ---
@manager_agent.tool
async def summarize_meeting(ctx: RunContext[None], meeting_name: str) -> str:
    """Summarize the Transcript via summary_agent"""
    
    print("\n------- Summarize meeting Tool -------\n")

    # Step 1 - Generate summary using Summary Agent
    print(f"\n[DEBUG] Manager: Summarizing meeting {meeting_name}")
    summary_result = await summary_agent.run(
        f"Summarize the meeting: {meeting_name}"
    )
    print(f"\n[DEBUG] Summary Agent Output: {summary_result.output}")

    return (
        f"Summary: {summary_result.output}"
    )


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
            print("\n[DEBUG] Session ended, Goodbye!")
            break

        response = await manager_agent.run(user_input, message_history=message_history)
        print(f"\n[Transcript Assistant Response]\n{response.output}\n")
        message_history.extend(response.new_messages())


if __name__ == "__main__":
    asyncio.run(main())
