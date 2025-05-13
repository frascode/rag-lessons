import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test():
    """
        sdfasdfasdfasdfasdf
    """
   

def ask_to_the_model(query):
    prompt = """You're a JSON Extractor. The user will pass actions that want to save in a todo list.For each query the user will provide. Do as follow:

            Output: {{
                "{action}": "action description",
                "date": date
            }}
            
            Example:
            User Query: "Domani voglio andare a pagare la bolletta."
            Output: {{
                "{action}": "Pagare la bolletta."
                "date": "Domani"
            }}
            
            User Query: "Giorno 10/05/2025 voglio andare al mare."
            Output: {{
                "{action}": "Voglio andare al mare."
                "date": "10/05/2025"
            }}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content


result = ask_to_the_model(
    "fare", "Giorno 25/12/2000 voglio andare allo stadio")
print(result)

result = ask_to_the_model(
    "non fare", "Giorno 25/05/2029 voglio andare dal medico")
print(result)
