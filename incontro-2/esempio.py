import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHAT_HISTOY = []

# Come si dichiarano le liste in python

# Esempio di Zero-shot prompting per analisi sentiment feedback cliente
def ask_to_the_model(query):

    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """Sei un assitente specializzato in linguaggi di programmazione. L'utente ti farà delle domande sul linguaggio e dovrai rispondere con uno stile molto formale.
             """},
            {"role": "user", "content": "\n".join(query)}
        ]
    )
    
    return response.choices[0].message.content



while True:
    query = input(">>>")
    CHAT_HISTOY.append(query)
    result = ask_to_the_model(CHAT_HISTOY)
    CHAT_HISTOY.append(result)
    print(result)
