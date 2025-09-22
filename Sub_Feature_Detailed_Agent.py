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
        "You are a sub features detail extractor.\n\n"
        "Task:\n"
        "Given a main feature transcript and SUB FEATURES, extract ONLY the exact lines from the transcript "
        "that are directly related to that sub features.\n\n"
        "Guidelines:\n"
        "- DO NOT summarize, rephrase, or invent text.\n"
        "- Copy the exact sentences/lines from the transcript verbatim.\n"
        "- Include all relevant discussions, decisions, and technical details word-for-word.\n"
        "- Exclude any unrelated or off-topic lines.\n"
        "- Output should be a clean, readable block of extracted transcript lines in plain text.\n"
    ),
)

# -------------------------------
# Extract Sub-Feature Details (with resume)
# -------------------------------
async def extract_sub_features_details(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """
    Reads sub_features.txt,
    creates folders for each main feature,
    and generates .txt files for each sub-feature (with hierarchy preserved).
    Uses only the main-feature transcript instead of the whole transcript.
    If process crashes, it resumes from where it left off.
    """

    print("\n------- Extract Detailed Features Tool -------\n")

    try:
        folder_name = meeting_name.replace(" ", "_")
        sub_features_path = os.path.join(folder_name, "sub_features.txt")

        # -------------------------------
        # Ensure sub_features.txt exists
        # -------------------------------
        if not os.path.exists(sub_features_path):
            return f"[DEBUG] No sub_features.txt found in {folder_name}"

        # -------------------------------
        # Parse sub_features.txt (preserve hierarchy)
        # -------------------------------
        with open(sub_features_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        main_feature = None
        sub_features_map = {}

        for line in lines:
            raw = line.rstrip("\n")
            stripped = raw.strip()

            if stripped and re.match(r"^\d+\)", stripped):  # main feature line
                main_feature = stripped.split(")", 1)[1].strip()
                sub_features_map[main_feature] = []
            elif stripped.startswith(("-", "–", "•")):
                # preserve indentation depth (count leading spaces)
                indent_level = len(raw) - len(raw.lstrip(" "))
                sub_features_map[main_feature].append((indent_level, stripped))

        # -------------------------------
        # Create folders + generate files
        # -------------------------------
        for main_feature, sub_features in sub_features_map.items():
            safe_main = re.sub(r'[^a-zA-Z0-9_-]', '_', main_feature.lower())
            main_folder = os.path.join(folder_name, safe_main)
            os.makedirs(main_folder, exist_ok=True)

            # Load only the transcript for this main feature
            main_feature_file = os.path.join(main_folder, f"{safe_main}.txt")
            if not os.path.exists(main_feature_file):
                print(f"[WARN] No main feature transcript found for {main_feature}, skipping...")
                continue

            with open(main_feature_file, "r", encoding="utf-8") as ft:
                main_feature_text = ft.read()

            if not sub_features:
                print(f"[DEBUG] No sub-features for {main_feature}, skipping.")
                continue

            # Group all sub-features into blocks by first-level entries ("- ...")
            feature_blocks = []
            current_block = []
            for indent, sub in sub_features:
                if indent <= 3 and sub.startswith("-"):  # new feature block
                    if current_block:
                        feature_blocks.append("\n".join(current_block))
                        current_block = []
                current_block.append(" " * indent + sub)
            if current_block:
                feature_blocks.append("\n".join(current_block))

            # -------------------------------
            # Process each feature block
            # -------------------------------
            for block in feature_blocks:
                block_title = block.split("\n", 1)[0].lstrip("-–• ").strip()
                safe_sub = re.sub(r'[^a-zA-Z0-9_-]', '_', block_title.lower())
                sub_file = os.path.join(main_folder, f"{safe_sub}.txt")

                # ✅ Skip if file already exists (resume mechanism)
                if os.path.exists(sub_file):
                    print(f"[RESUME] Skipping {block_title}, file already exists.")
                    continue

                try:
                    prompt_input = (
                        f"Transcript (for this main feature only):\n{main_feature_text}\n\n"
                        f"Feature Block:\n{block}"
                    )

                    response = await detailed_agent.run(prompt_input)
                    feature_details: FeatureDetails = response.output

                    with open(sub_file, "w", encoding="utf-8") as f:
                        f.write(f"Main Feature: {main_feature}\n")
                        f.write(f"Feature Block:\n{block}\n")
                        f.write("=" * (20 + len(block_title)) + "\n\n")
                        f.write(feature_details.details)

                    print(f"[DEBUG] Saved details for {block_title} -> {sub_file}")

                except Exception as block_err:
                    print(f"[ERROR] Failed on block {block_title}: {block_err}")
                    # Continue with next block instead of crashing
                    continue

        return f"Detailed sub-feature files created inside {folder_name}/<main_feature> folders"

    except Exception as e:
        return f"[ERROR] {str(e)}"


