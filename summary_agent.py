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
model = GoogleModel("gemini-1.5-flash", provider=provider)

# -------------------------------
# Models
# -------------------------------
class Summary(BaseModel):
    summary: str

# -------------------------------
# Main Agent Definition
# -------------------------------
summary_agent = Agent(
    model,
    output_type=Summary,
    system_prompt=(
        "You are a meeting summarization assistant.\n"
        "Your task is to generate a summary of meeting transcripts."
    ),
)

# -------------------------------
# Summary Tool
# -------------------------------
@summary_agent.tool
async def generate_meeting_summary(ctx: RunContext[None], meeting_name: str) -> str:
    """
    Generates a short summary of the meeting transcript identified by its unique meeting name,
    saves it in database.json and also writes it into <meeting_name>_summary.txt.
    """
    
    print("\n------- Generate Meeting Summary Tool -------\n")
    
    try:
        with open("database.json", "r") as db_file:
            db_data = json.load(db_file)

        # Find the meeting
        meeting = next((m for m in db_data if m.get("title") == meeting_name), None)
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
                f"You are summarizing a meeting transcript.\n"
                f"The discussion in this meeting was about an app named '{meeting_name}'.\n"
                "Summarize all the important points discussed clearly and concisely.\n"
                "Do not add new information or opinions."
            ),
        )

        # Run contextual agent → returns Summary model
        response = await contextual_summary_agent.run(transcript_text)
        summary_output: Summary = response.output

        # Save summary to DB
        meeting["summary"] = summary_output.summary
        with open("database.json", "w") as db_file:
            json.dump(db_data, db_file, indent=4)

        # Also save to a new file
        file_name = f"{meeting_name.replace(' ', '_')}_summary.txt"
        with open(file_name, "w") as summary_file:
            summary_file.write(f"Meeting: {meeting_name}\n")
            summary_file.write("=" * (9 + len(meeting_name)) + "\n\n")
            summary_file.write(summary_output.summary)

        return f"\n\nSummary for meeting '{meeting_name}' saved in database.json and {file_name}"

    except FileNotFoundError:
        return "[ERROR] Database file not found."
    except json.JSONDecodeError:
        return "[ERROR] Error decoding database JSON."
