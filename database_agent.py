import json
import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

load_dotenv()

# (From Step 2: Define Shared Models)
class Transcript(BaseModel):
    id: int
    filepath: str
    title: str
    text: str
    created_at: str
    updated_at: str

# Agent + Provider Setup
provider = GoogleProvider(api_key=os.getenv("GOOGLE_API_KEY"))
model = GoogleModel("gemini-1.5-flash", provider=provider)

# Main agent for database operations
agent = Agent(
    model,
    system_prompt="You are a database agent. Your job is to save and update transcripts.",
)
@agent.tool
async def save_transcript(ctx: RunContext[None], file_path: str, meeting_name: str) -> str:
    """
    Reads a transcript from a file, creates a new transcript record,
    and saves it to the database.
    """
    
    print("\n------- Save Transcript Tool -------\n")

    # Read transcript file
    with open(file_path, 'r') as f:
        content = f.read()

    # Ensure database.json exists
    if not os.path.exists("database.json"):
        print("\n[WARN] database.json not found, creating a new one")
        with open("database.json", "w") as db_file:
            json.dump([], db_file)

    # Load database safely
    with open("database.json", "r+") as db_file:
        try:
            db_data = json.load(db_file)
        except json.JSONDecodeError:
            print("[WARN] database.json is empty or invalid, reinitializing as empty list")
            db_data = []

        # Check for existing transcript by meeting name
        for record in db_data:
            if record.get("title") == meeting_name:
                record["text"] = content
                record["filepath"] = file_path
                record["updated_at"] = datetime.datetime.now().isoformat()

                db_file.seek(0)
                db_file.truncate()
                json.dump(db_data, db_file, indent=4)
                return f"Transcript '{meeting_name}' updated successfully."

        # Create new record
        new_transcript = Transcript(
            id=len(db_data) + 1,
            filepath=file_path,
            title=meeting_name,
            text=content,
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat(),
        )

        db_data.append(new_transcript.model_dump())

        db_file.seek(0)
        db_file.truncate()
        json.dump(db_data, db_file, indent=4)

    return f"Transcript '{meeting_name}' saved successfully."
