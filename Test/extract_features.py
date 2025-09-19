import os
import json
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
class MainFeatures(BaseModel):
    features: list[str]  # list of main big-picture features

# -------------------------------
# Setup Google model provider
# -------------------------------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

# -------------------------------
# Extract Feature Agent
# -------------------------------
extract_feature_agent = Agent(
    model,
    output_type=MainFeatures,
    system_prompt=(
        "You are an assistant that extracts ONLY the MAIN BIG-PICTURE FEATURES "
        "from a meeting transcript.\n\n"
        "Guidelines:\n"
        "- A main feature = one broad, high-level capability or functionality.\n"
        "- Keep each feature SHORT and ABSTRACT (no details, no examples, no subpoints).\n"
        "- Do NOT include sub-features, technical details, implementation notes, or action items.\n"
        "- Merge overlapping ideas into one unified main feature.\n"
        "- Limit output to a clean list of distinct, top-level features only.\n\n"
    ),
)


async def extract_main_features(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """
    Finds the meeting transcript in database.json using file_path,
    extracts ONLY main big-picture features,
    and saves them in main_features.txt
    """

    print("\n------- Extract Main Features Tool -------\n")

    try:
        folder_name = meeting_name.replace(" ", "_")
        db_path = os.path.join(folder_name, "database.json")
        
        with open(db_path, "r") as db_file:
            db_data = json.load(db_file)

        # Find meeting by file_path
        meeting = next((m for m in db_data if m.get("filepath") == file_path), None)
        if not meeting:
            return f"[DEBUG] No meeting found for file path: {file_path}"

        transcript_text = meeting.get("text", "")
        if not transcript_text:
            return f"[DEBUG] Meeting at '{file_path}' has no transcript text."

        # Step 1 - Extract main features
        response = await extract_feature_agent.run(transcript_text)
        main_features: MainFeatures = response.output
        features_list = main_features.features

        print(f"\n[DEBUG] Extracted Main Features: {features_list}\n")

        if not features_list:
            return "[DEBUG] No main features found in transcript."
        
        # Step 2 - Save main_features.txt
        feature_path = os.path.join(folder_name, "main_features.txt")
        with open(feature_path, "w") as features_file:
            features_file.write("Extracted Main Features:\n\n")
            for feat in features_list:
                features_file.write(f"- {feat}\n")
                
        print(f"\n[DEBUG] Main features saved to {feature_path}\n")

        return f"Main features extracted: {len(features_list)}\nSaved in {feature_path}"

    except Exception as e:
        return f"[ERROR] {str(e)}"
