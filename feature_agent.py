import os
import json
import re
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# Models
# -------------------------------
class MeetingFeatures(BaseModel):
    features: list[str]  # list of extracted features

class FeatureDetails(BaseModel):
    feature: str
    details: str

# -------------------------------
# Setup Google model provider
# -------------------------------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

# -------------------------------
# Feature Agents
# -------------------------------
# 1) Extract the features list
feature_agent = Agent(
    model,
    system_prompt=(
        "You are a meeting feature extraction assistant.\n"
    ),
)

# 2) Extract details for a specific feature
feature_detail_agent = Agent(
    model,
    output_type=FeatureDetails,
    system_prompt=(
        "You are a meeting detail extractor.\n"
        "Given a meeting transcript and a feature, extract all details, decisions, and discussions "
        "related only to that feature.\n"
        "Return them as a clean summary."
    ),
)

# 3) Extract Features
extract_feature_agent = Agent(
    model,
    output_type=MeetingFeatures,
    system_prompt = """
                    You are an assistant that analyzes meeting transcripts.
                    Your ONLY task is to identify and extract **meeting features**.

                    Do NOT include:
                    - Discussion points
                    - Action items
                    - Summaries
                    - Irrelevant details

                    Only return the features list.

                    Answer strictly with features only, nothing else.
                """
)

# -------------------------------
# Feature Extraction Tool
# -------------------------------
@feature_agent.tool
async def extract_meeting_features(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """
    Finds the meeting transcript in database.json using file_path,
    extracts features, saves them in features.txt,
    and also creates a file for each feature with its detailed discussion.
    """

    print("\n------- Extract Meeting Features Tool -------\n")

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
            return f"[DEBUG] Meeting at '{file_path}' has no transcript text for feature extraction."

        # Step 1 - Extract features
        response = await extract_feature_agent.run(transcript_text)
        features_output: MeetingFeatures = response.output
        features_list = features_output.features
        
        print(f"\n[DEBUG] Extracted Features: {features_list}\n")

        if not features_list:
            return "[DEBUG] No features found in transcript."
        
        # Step 2 - Save features.txt
        feature_path = os.path.join(folder_name, "features.txt")
        with open(feature_path, "w") as features_file:
            features_file.write("Extracted Meeting Features:\n\n")
            for feat in features_list:
                features_file.write(f"- {feat}\n")
                
        print(f"\n[DEBUG] Features saved to {feature_path}\n")

        # Step 3 - Create individual feature files with details
        for feature in features_list:
            detail_response = await feature_detail_agent.run(
                f"Feature: {feature}\n\nTranscript:\n{transcript_text}"
            )
            feature_details: FeatureDetails = detail_response.output

            # sanitize feature name for filename
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', feature.lower())
            feature_file = os.path.join(folder_name, f"{safe_name}.txt")

            with open(feature_file, "w") as f:
                f.write(f"Feature: {feature}\n")
                f.write("=" * (10 + len(feature)) + "\n\n")
                f.write(feature_details.details)
            
        print(f"\n[DEBUG] Individual feature files created in {folder_name}\n")

        return (
            f"Features extracted: {len(features_list)}\n"
            f"- Saved in {features_file}\n"
            f"- Individual feature files created in {folder_name}"
        )

    except Exception as e:
        return f"[ERROR] {str(e)}"
