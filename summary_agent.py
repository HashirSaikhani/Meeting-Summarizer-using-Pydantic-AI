import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# Setup Google model provider
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

# -------------------------------
# Models
# -------------------------------
class Summary(BaseModel):
    summary: str

# -------------------------------
# Summary Tool
# -------------------------------
async def generate_meeting_summary(ctx: RunContext[None], meeting_name: str, file_path) -> str:
    """
    Generates a short summary of the meeting transcript identified by its unique meeting name,
    saves it in database.json and also writes it into <meeting_name>_summary.txt.
    """
    
    print("\n------- Generate Meeting Summary Tool -------\n")
    
    try:
        
        folder_name = meeting_name.replace(" ", "_")
        db_path = os.path.join(folder_name, "database.json")
        
        with open(db_path, "r") as db_file:
            db_data = json.load(db_file)

        # Find the meeting
        meeting = next((rec for rec in db_data if rec.get("filepath") == file_path), None)
        if not meeting:
            return f"[DEBUG] Meeting '{meeting_name}' not found."

        transcript_text = meeting.get("text", "")
        if not transcript_text:
            return f"[DEBUG] Meeting '{meeting_name}' has no transcript text to summarize."

        # Create a contextual sub-agent for this specific summary
        contextual_summary_agent = Agent(
            model,
            output_type=Summary,
            system_prompt=(
                f"You are summarizing the following meeting transcript for '{meeting_name}'.\n"
                f"Summarize all the important points discussed clearly and concisely.\n"
                f"Do not add new information or opinions."
            ),
            tools=[] 
        )

        # Run contextual agent → returns Summary model
        response = await contextual_summary_agent.run(transcript_text)
        summary_output: Summary = response.output

       # Save summary into the same folder as database.json
        summary_path = os.path.join(folder_name, "summary.txt")
        with open(summary_path, "w") as summary_file:
            summary_file.write(f"Meeting: {meeting_name}\n")
            summary_file.write("=" * (9 + len(meeting_name)) + "\n\n")
            summary_file.write(summary_output.summary)

        return f"\n\nSummary for meeting '{meeting_name}' saved in {file_path}"

    except FileNotFoundError:
        return "[ERROR] Database file not found."
    except json.JSONDecodeError:
        return "[ERROR] Error decoding database JSON."
