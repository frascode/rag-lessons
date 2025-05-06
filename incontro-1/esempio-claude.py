from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "Quali sono le best practice per un CRM?"}
    ]
)

print(message.content[0].text)