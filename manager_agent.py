import asyncio
import os
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# Import sub-agents
from database_agent import save_transcript
from summary_agent import generate_meeting_summary
from Extract_Main_Features_Agent import extract_main_features
from Main_Feature_Detailed_Agent import extract_main_features_details
from Extract_Sub_Features_Agent import extract_sub_features
from Sub_Feature_Detailed_Agent import extract_sub_features_details

# --------- Load ENV ---------
load_dotenv()

# --------- Agent + Provider Setup ---------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

manager_agent = Agent(
    model=model,
    system_prompt=(
        "You are the Manager Agent for the Meeting System.\n\n"
        "Your workflow is strictly sequential:\n"
        "1. Save meeting → 2. Summarize meeting → 3. Extract main features "
        "→ 4. Generate main feature details → 5. Extract sub-features → 6. Generate sub-feature details.\n\n"
        "Rules:\n"
        "- Never skip or change the order.\n"
        "- Keep responses short and clear."
    ),
)

# --- Tool 1: Save Meeting ---
@manager_agent.tool
async def save_meeting(ctx: RunContext[None], file_path: str) -> str:
    """Save or update meeting via database_agent"""
    print("\n------- Manager Save Meeting Tool -------\n")
    db_result = await save_transcript(ctx, file_path)
    return f"✅ Meeting saved.\nDatabase Response: {db_result}\n"

# --- Tool 2: Summarize Meeting ---
@manager_agent.tool
async def summarize_meeting(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """Summarize the meeting via summary_agent"""
    print("\n------- Manager Summarize Meeting Tool -------\n")
    summary_result = await generate_meeting_summary(ctx, meeting_name, file_path)
    return f"✅ Summary generated.\n{summary_result}"

# --- Tool 3: Extract Main Features ---
@manager_agent.tool
async def extract_main(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """Extract main features"""
    print("\n------- Manager Extract Main Features Tool -------\n")
    result = await extract_main_features(ctx, meeting_name, file_path)
    return f"✅ Main features extracted.\n{result}"

# --- Tool 4: Main Feature Details ---
@manager_agent.tool
async def extract_main_details(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """Generate detailed files for main features"""
    print("\n------- Manager Main Feature Details Tool -------\n")
    result = await extract_main_features_details(ctx, meeting_name, file_path)
    return f"✅ Main feature details created.\n{result}"

# --- Tool 5: Extract Sub-Features ---
@manager_agent.tool
async def extract_sub(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """Extract sub-features"""
    print("\n------- Manager Extract Sub-Features Tool -------\n")
    result = await extract_sub_features(ctx, meeting_name, file_path)
    return f"✅ Sub-features extracted.\n{result}"

# --- Tool 6: Sub-Feature Details ---
@manager_agent.tool
async def extract_sub_details(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """Generate detailed files for sub-features"""
    print("\n------- Manager Sub-Feature Details Tool -------\n")
    result = await extract_sub_features_details(ctx, meeting_name, file_path)
    return f"✅ Sub-feature details created.\n{result}"

# --- Main Interactive Loop ---
async def main():
    print("\n[DEBUG] Manager Agent interactive session started.")
    print("Type 'exit' to quit.\n")

    message_history = []
    response = await manager_agent.run("Hi!", message_history=message_history)
    print(f"{response.output}\n")
    message_history.extend(response.new_messages())

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("\n[DEBUG] Session ended, Goodbye!\n")
            break

        response = await manager_agent.run(user_input, message_history=message_history)
        print(f"\n[Meeting Assistant Response]\n{response.output}\n")
        message_history.extend(response.new_messages())

if __name__ == "__main__":
    asyncio.run(main())
