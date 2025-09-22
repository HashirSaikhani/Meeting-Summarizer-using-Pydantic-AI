import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import re

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# -------------------------------
# Models
# -------------------------------
class HierarchicalFeatures(BaseModel):
    features: list[str]  # hierarchical features with indentation

# -------------------------------
# Setup Google model provider
# -------------------------------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

# -------------------------------
# Hierarchical Sub-Feature Agent
# -------------------------------
extract_hierarchical_agent = Agent(
    model,
    output_type=HierarchicalFeatures,
    system_prompt=(
        
        "You are an assistant that extracts ONLY coding-related sub-features(implementation-level functionalities) for a given MAIN feature from a meeting transcript.\n\n"
        
        "Instruction:\n"
        "- NEVER repeat or re-list the main feature itself.\n"
        "- ONLY generate sub-features that belong directly under the given main feature.\n"
        "- If no sub-features exist, only return the main feature line with no children.\n"
        "- Avoid redundancy: if multiple points belong to the same concept, group them together.\n"
        "- Do NOT include discussion points, action items, summaries, or irrelevant details.\n"
        "- Exclude high-level ideas that have no direct coding implication.\n\n"
        
        "- Use a clear structured hierarchy:\n"
           "1) Main Feature\n"
            "   - Sub-feature (if any)\n"
            "        -- sub-sub-feature (if any)\n"
            "           --- sub-sub-sub-feature (if any)\n"
            "               ---- sub-sub-sub-sub-feature (if any)\n"
            "   - Sub-feature (if any)\n"
            "2) Next Main Feature\n"
               "- Sub-feature (if any)\n"
       
    ),
)


async def extract_sub_features(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """
    Incrementally extracts hierarchical sub-features for each main feature.
    Uses the detailed transcript text stored per feature instead of the whole transcript.
    Progress is saved after each feature so process can resume safely if interrupted.
    """

    print("\n------- Extract Hierarchical Sub-Features Tool -------\n")

    try:
        folder_name = meeting_name.replace(" ", "_")
        main_features_path = os.path.join(folder_name, "main_features.txt")
        sub_features_path = os.path.join(folder_name, "sub_features.txt")

        # -------------------------------
        # Load main features
        # -------------------------------
        if not os.path.exists(main_features_path):
            return f"[DEBUG] No main_features.txt found in {folder_name}"

        with open(main_features_path, "r") as f:
            lines = f.readlines()

        main_features = [line.strip("- ").strip() for line in lines if line.startswith("- ")]
        print(f"[DEBUG] Found {len(main_features)} main features to process.")

        # -------------------------------
        # Determine which features already extracted
        # -------------------------------
        completed_features = set()
        if os.path.exists(sub_features_path):
            with open(sub_features_path, "r", encoding="utf-8") as f:
                content = f.read()
                for feature in main_features:
                    if feature in content:
                        completed_features.add(feature)

        # -------------------------------
        # Open master sub-features file in append mode
        # -------------------------------
        with open(sub_features_path, "a", encoding="utf-8") as f:
            if os.stat(sub_features_path).st_size == 0:
                f.write("Extracted Hierarchical Features:\n\n")

            for idx, main_feature in enumerate(main_features, start=1):
                if main_feature in completed_features:
                    print(f"[DEBUG] Skipping already completed feature: {main_feature}")
                    continue

                print(f"\n[DEBUG] Processing main feature {idx}) {main_feature}")

                # -------------------------------
                # Load transcript relevant to this feature
                # -------------------------------
                safe_feature = re.sub(r'[^a-zA-Z0-9_-]', '_', main_feature.lower())
                feature_file = os.path.join(folder_name, safe_feature, f"{safe_feature}.txt")

                if not os.path.exists(feature_file):
                    print(f"[WARN] No detailed transcript file found for {main_feature}, skipping...")
                    continue

                with open(feature_file, "r", encoding="utf-8") as ft:
                    feature_transcript_text = ft.read()

                # -------------------------------
                # Build prompt
                # -------------------------------
                prompt_input = (
                    f"Transcript (related to this feature only):\n{feature_transcript_text}\n\n"
                    f"Main Feature to expand: {idx}) {main_feature}\n\n"
                    "Extract ONLY the sub-features of this feature in proper hierarchical format."
                )

                # -------------------------------
                # Run sub-feature extraction
                # -------------------------------
                result = await extract_hierarchical_agent.run(prompt_input)

                if result.output and result.output.features:
                    sub_features_text = "\n".join(result.output.features)
                    f.write(sub_features_text + "\n\n")
                else:
                    f.write(f"{idx}) {main_feature}\n\n")

                f.flush()
                print(f"[DEBUG] Appended {main_feature} → {sub_features_path}")

        print(f"\n[DEBUG] Hierarchical sub-features saved to {sub_features_path}\n")
        return f"Hierarchical sub-features extracted and saved in {sub_features_path}"

    except Exception as e:
        return f"[ERROR] {str(e)}"

