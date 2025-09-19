import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# Import the three agents
from extract_features import extract_main_features
from extract_subfeatures import extract_sub_features
from detailed_agent import extract_detailed_features

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# Agent + Provider Setup
# -------------------------------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

# -------------------------------
# Temp Manager Agent
# -------------------------------
temp_manager = Agent(
    model=model,
    system_prompt=(
        "You are the Temp Manager Agent for testing purposes.\n\n"
        "Your responsibility:\n"
        "1. Call the main feature extractor.\n"
        "2. Then call the sub-feature extractor.\n"
        "3. Finally call the detailed extractor.\n"
        "Run them in this sequence whenever asked to extract meeting features."
    ),
)

# -------------------------------
# Tool: Sequential Feature Extraction
# -------------------------------
@temp_manager.tool
async def extract_all_features(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """
    Runs the 3-step pipeline:
    1. Extract main features
    2. Extract sub-features
    3. Extract detailed sub-feature files
    """

    print("\n------- Temp Manager Full Pipeline -------\n")
    print(f"[DEBUG] Extracting all features for meeting '{meeting_name}' from {file_path}\n")

    # # Step 1: Extract main features
    # step1 = await extract_main_features(ctx, meeting_name, file_path)
    # print(f"[DEBUG] Step 1 (Main Features): {step1}\n")

    # Step 2: Extract sub-features
    step2 = await extract_sub_features(ctx, meeting_name, file_path)
    print(f"[DEBUG] Step 2 (Sub-Features): {step2}\n")

    # Step 3: Extract detailed features
    step3 = await extract_detailed_features(ctx, meeting_name, file_path)
    print(f"[DEBUG] Step 3 (Detailed Features): {step3}\n")

    return (
        f"✅ Full pipeline completed for meeting '{meeting_name}'.\n\n"
        f"Step 1: {step1}\n"
        f"Step 2: {step2}\n"
        f"Step 3: {step3}"
    )

# -------------------------------
# Main Interactive Loop
# -------------------------------
async def main():
    print("\n[DEBUG] Temp Manager Agent interactive session started.")
    print("Type 'exit' to quit.\n")

    message_history = []

    # Initial pipeline run
    response = await temp_manager.run(
        "please extract features for meeeting named TEMP and path is Examples/2.txt",
        message_history=message_history
    )
    print(f"{response.output}\n")
    message_history.extend(response.new_messages())

    # Uncomment if you want interactive mode
    # while True:
    #     user_input = input("You: ").strip()
    #     if user_input.lower() in {"exit", "quit"}:
    #         print("\n[DEBUG] Session ended, Goodbye!\n")
    #         break
    #
    #     response = await temp_manager.run(user_input, message_history=message_history)
    #     print(f"\n[Temp Manager Response]\n{response.output}\n")
    #     message_history.extend(response.new_messages())

if __name__ == "__main__":
    asyncio.run(main())
