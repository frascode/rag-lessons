import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Sei un assistente CRM esperto."},
        {"role": "user", "content": "Cos'è un lead scoring?"}
    ]
)

print(response.choices[0].message.content)