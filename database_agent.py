import json
import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
import shutil


load_dotenv()

# -------------------------------
# Models
# -------------------------------
class Transcript(BaseModel):
    id: int
    filepath: str
    title: str | None = None
    text: str
    created_at: str
    updated_at: str

class MeetingName(BaseModel):
    name: str

# -------------------------------
# Agent Setup
# -------------------------------
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-2.5-flash", provider=provider)

# Naming tool agent
naming_tool_agent = Agent(
    model,
    output_type=MeetingName,
    system_prompt=(
        "You are a meeting naming assistant. "
        "Your task is to analyze the meeting transcript and assign a short, clear, descriptive name "
        "that reflects the main purpose or topic of the meeting. "
        "Do not explain, just return the name."
    ),
    model_settings=GoogleModelSettings(temperature=0.1)
)

# -------------------------------
# Save Transcript + Generate Name
# -------------------------------
async def save_transcript(ctx: RunContext[None], file_path: str) -> str:
    """
    Reads a transcript file, saves it into database.json,
    generates a descriptive meeting title only if the meeting is new,
    and returns status + title.
    """

    print("\n------- Save Transcript Tool -------\n")

    # Step 1 - Read transcript file
    if not os.path.exists(file_path):
        return f"[ERROR] File '{file_path}' not found."

    with open(file_path, "r") as f:
        content = f.read()

    # Step 2 - Ensure database.json exists
    if not os.path.exists("database.json"):
        print("\n[WARN] database.json not found, creating a new one\n")
        with open("database.json", "w") as db_file:
            json.dump([], db_file)

    # Step 3 - Load database safely
    with open("database.json", "r+") as db_file:
        try:
            db_data = json.load(db_file)
        except json.JSONDecodeError:
            print("[\nWARN] database.json invalid, reinitializing\n")
            db_data = []

        # Step 4 - Check if record already exists
        meeting = next((rec for rec in db_data if rec.get("filepath") == file_path), None)

        if meeting:
            # Update transcript content (do NOT change title)
            meeting["text"] = content
            meeting["updated_at"] = datetime.datetime.now().isoformat()
            assigned_title = meeting.get("title", "Untitled Meeting")
            print(f"\n[DEBUG] Updating existing record for {file_path}\n")

        else:
            # Create new transcript object
            transcript = Transcript(
                id=len(db_data) + 1,
                filepath=file_path,
                text=content,
                created_at=datetime.datetime.now().isoformat(),
                updated_at=datetime.datetime.now().isoformat(),
                title=None,  # will be assigned by name agent
            )
            db_data.append(transcript.model_dump())
            print(f"\n[DEBUG] New record created for {file_path}\n")

            # Step 5 - Generate meeting title using naming agent (only for new)
            response = await naming_tool_agent.run(content)
            meeting_name_output: MeetingName = response.output
            assigned_title = meeting_name_output.name

            # Update record
            db_data[-1]["title"] = assigned_title
            db_data[-1]["updated_at"] = datetime.datetime.now().isoformat()

            # Move DB to meeting folder
            move_db_to_meeting_folder(assigned_title)

        # Step 6 - Save DB back
        db_file.seek(0)
        db_file.truncate()
        json.dump(db_data, db_file, indent=4)

    return f"Transcript saved. Meeting title: {assigned_title}"



def move_db_to_meeting_folder(meeting_name: str):
    """
    Create a folder named after the meeting and move database.json into it.
    """
    folder_name = meeting_name.replace(" ", "_")  # replace spaces with underscores
    os.makedirs(folder_name, exist_ok=True)

    target_path = os.path.join(folder_name, "database.json")

    try:
        shutil.move("database.json", target_path)
        print(f"\n[INFO] database.json moved to: {target_path}\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to move database.json: {e}\n")
