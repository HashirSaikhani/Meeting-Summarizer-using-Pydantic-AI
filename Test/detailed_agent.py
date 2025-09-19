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
# Models
# -------------------------------
class FeatureDetails(BaseModel):
    subfeature: str
    details: str

# -------------------------------
# Setup Google model provider
# -------------------------------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

# -------------------------------
# Detailed Agent
# -------------------------------
detailed_agent = Agent(
    model,
    output_type=FeatureDetails,
    system_prompt=(
        "You are a feature detail extractor.\n\n"
        "Given a meeting transcript and ONE sub-feature, extract all related discussions, "
        "decisions, and technical details.\n\n"
        "Guidelines:\n"
        "- Stay focused only on that specific sub-feature.\n"
        "- Ignore irrelevant discussions.\n"
        "- Return a clean, readable summary in plain text."
    ),
)

async def extract_detailed_features(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """
    Reads sub_features.txt,
    creates folders for each main feature,
    and generates .txt files for each sub-feature with extracted details.
    """

    print("\n------- Extract Detailed Features Tool -------\n")

    try:
        folder_name = meeting_name.replace(" ", "_")
        db_path = os.path.join(folder_name, "database.json")
        sub_features_path = os.path.join(folder_name, "sub_features.txt")

        # Load transcript
        with open(db_path, "r") as db_file:
            db_data = json.load(db_file)

        meeting = next((m for m in db_data if m.get("filepath") == file_path), None)
        if not meeting:
            return f"[DEBUG] No meeting found for file path: {file_path}"

        transcript_text = meeting.get("text", "")
        if not transcript_text:
            return f"[DEBUG] Meeting at '{file_path}' has no transcript text."

        # Ensure sub_features.txt exists
        if not os.path.exists(sub_features_path):
            return f"[DEBUG] No sub_features.txt found in {folder_name}"

        # Parse sub_features.txt
        with open(sub_features_path, "r") as f:
            lines = f.readlines()

        main_feature = None
        sub_features_map = {}

        for line in lines:
            line = line.strip()
            if line and re.match(r"^\d+\)", line):  # main feature line
                main_feature = line.split(")", 1)[1].strip()
                sub_features_map[main_feature] = []
            elif line.startswith("-") or line.startswith("–") or line.startswith("•") or line.startswith("   -"):
                sub = line.replace("-", "").strip()
                if main_feature:
                    sub_features_map[main_feature].append(sub)

        # Create folders + files
        for main_feature, sub_features in sub_features_map.items():
            safe_main = re.sub(r'[^a-zA-Z0-9_-]', '_', main_feature.lower())
            main_folder = os.path.join(folder_name, safe_main)
            os.makedirs(main_folder, exist_ok=True)

            if not sub_features:
                print(f"[DEBUG] No sub-features for {main_feature}, skipping.")
                continue

            for sub in sub_features:
                prompt_input = f"Transcript:\n{transcript_text}\n\nSub-Feature:\n{sub}"
                response = await detailed_agent.run(prompt_input)
                feature_details: FeatureDetails = response.output

                safe_sub = re.sub(r'[^a-zA-Z0-9_-]', '_', sub.lower())
                file_path = os.path.join(main_folder, f"{safe_sub}.txt")

                with open(file_path, "w") as f:
                    f.write(f"Main Feature: {main_feature}\n")
                    f.write(f"Sub-Feature: {sub}\n")
                    f.write("=" * (15 + len(sub)) + "\n\n")
                    f.write(feature_details.details)

                print(f"[DEBUG] Saved details for {sub} -> {file_path}")

        return f"Detailed sub-feature files created inside {folder_name}/<main_feature> folders"

    except Exception as e:
        return f"[ERROR] {str(e)}"
