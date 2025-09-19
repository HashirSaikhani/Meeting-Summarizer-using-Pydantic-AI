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
        
        "You are an assistant that extracts sub features (implementation wise) from meeting transcripts."

        "Instructions:"
        "- Focus only on essential, meaningful sub features (implementation wise) if any"

        "- Avoid redundancy: if multiple points belong to the same concept, group them together."
        "- Use a clear structured hierarchy:"
           " 1) Main Feature"
            "   - Sub-feature (optional)"
            "        -- sub-sub-feature (optional)"
            "           --- sub-sub-sub-feature (optional)"
            "   - Sub-feature (optional)"
            "2) Next Main Feature"
    
               "- Sub-feature (optional)"
        "- Do NOT include discussion points, action items, summaries, or irrelevant details."
    ),
)

# -------------------------------
# Extract Sub-Features (One-shot hierarchy)
# -------------------------------
async def extract_sub_features(ctx: RunContext[None], meeting_name: str, file_path: str) -> str:
    """
    Reads main features from main_features.txt,
    extracts hierarchical sub-features (multi-level in one shot),
    and writes them as a tree structure in sub_features.txt
    """

    print("\n------- Extract Hierarchical Sub-Features Tool -------\n")

    try:
        folder_name = meeting_name.replace(" ", "_")
        db_path = os.path.join(folder_name, "database.json")
        main_features_path = os.path.join(folder_name, "main_features.txt")
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

        # Load main features
        if not os.path.exists(main_features_path):
            return f"[DEBUG] No main_features.txt found in {folder_name}"

        with open(main_features_path, "r") as f:
            lines = f.readlines()

        main_features = [line.strip("- ").strip() for line in lines if line.startswith("- ")]

        all_results = {}

        # Extract hierarchical structure for each main feature
        for main_feature in main_features:
            prompt_input = f"Transcript:\n{transcript_text}\n\nMain Feature:\n{main_feature}"
            response = await extract_hierarchical_agent.run(prompt_input)
            subfeatures: HierarchicalFeatures = response.output
            all_results[main_feature] = subfeatures.features

            print(f"[DEBUG] {main_feature} -> extracted hierarchy")

        # Save results
        with open(sub_features_path, "w") as f:
            f.write("Extracted Hierarchical Features:\n\n")
            for i, (main_feature, subs) in enumerate(all_results.items(), start=1):
                f.write(f"{i}) {main_feature}\n")
                if subs:
                    for line in subs:
                        f.write(f"{line}\n")
                else:
                    f.write("   (No sub-features found)\n")
                f.write("\n")

        print(f"\n[DEBUG] Hierarchical sub-features saved to {sub_features_path}\n")
        return f"Hierarchical sub-features extracted and saved in {sub_features_path}"

    except Exception as e:
        return f"[ERROR] {str(e)}"
