import os
import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# Pydantic Model
# -------------------------------
class FeatureDetails(BaseModel):
    details: str

# -------------------------------
# Setup Google model provider
# -------------------------------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

# -------------------------------
# Agent for extracting detailed info
# -------------------------------
detailed_agent = Agent(
    model,
    output_type=FeatureDetails,
    system_prompt=(
        "You are a main feature detail extractor.\n\n"
        "Task:\n"
        "Given a meeting transcript and ONE MAIN FEATURE, extract ONLY the exact lines from the transcript "
        "that are directly related to that feature.\n\n"
        "Guidelines:\n"
        "- DO NOT summarize, rephrase, or invent text.\n"
        "- Copy the exact sentences/lines from the transcript verbatim.\n"
        "- Include all relevant discussions, decisions, and technical details word-for-word.\n"
        "- Exclude any unrelated or off-topic lines.\n"
        "- Output should be a clean, readable block of extracted transcript lines in plain text.\n"
    ),
)


# -------------------------------
# Extract Detailed Features
# -------------------------------
async def extract_main_features_details(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """
    Reads main_features.txt,
    creates a folder for each main feature,
    and generates a .txt file with extracted details about that feature.
    """

    print("\n---------------- Extract Detailed Features Tool 2 called ----------------------\n")

    try:
        folder_name = meeting_name.replace(" ", "_")
        db_path = os.path.join(folder_name, "database.json")
        main_features_path = os.path.join(folder_name, "main_features.txt")

        # -------------------------------
        # Load transcript
        # -------------------------------
        with open(db_path, "r") as db_file:
            db_data = json.load(db_file)

        meeting = next((m for m in db_data if m.get("filepath") == file_path), None)
        if not meeting:
            return f"[DEBUG] No meeting found for file path: {file_path}"

        transcript_text = meeting.get("text", "")
        if not transcript_text:
            return f"[DEBUG] Meeting at '{file_path}' has no transcript text."

        # -------------------------------
        # Load main features
        # -------------------------------
        if not os.path.exists(main_features_path):
            return f"[DEBUG] No main_features.txt found in {folder_name}"

        with open(main_features_path, "r") as f:
            lines = f.readlines()

        main_features = [line.strip("- ").strip() for line in lines if line.startswith("- ")]
        print(f"[DEBUG] Found {len(main_features)} main features.")

        # -------------------------------
        # Process each main feature
        # -------------------------------
        for feature in main_features:
            safe_feature = re.sub(r'[^a-zA-Z0-9_-]', '_', feature.lower())
            feature_folder = os.path.join(folder_name, safe_feature)
            os.makedirs(feature_folder, exist_ok=True)

            print(f"\n[DEBUG] Processing feature: {feature}")

            # Build prompt
            prompt_input = f"Transcript:\n{transcript_text}\n\nMain Feature:\n{feature}"

            # Run agent
            response = await detailed_agent.run(prompt_input)
            feature_details: FeatureDetails = response.output

            # Save details
            file_path = os.path.join(feature_folder, f"{safe_feature}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Main Feature: {feature}\n")
                f.write("=" * (15 + len(feature)) + "\n\n")
                f.write(feature_details.details)

            print(f"[DEBUG] Saved details for {feature} -> {file_path}")

        return f"Detailed feature files created inside {folder_name}/<feature_name> folders"

    except Exception as e:
        return f"[ERROR] {str(e)}"
