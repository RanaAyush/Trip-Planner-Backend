import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("HF_TOKEN")
NER_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
CHAT_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

def query_huggingface(prompt, model_url, max_tokens=300, temperature=0.3):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }

    response = requests.post(model_url, headers=headers, json=payload)
    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return None

    result_text = response.json()[0]['generated_text']
    return result_text

def replyFollowingQuerry(context,query):
    prompt = f"""
### Task:
Based on the **Context** (previous question) and **Query** (user's response) to it, return a valid JSON object that fills exactly **one** of the following fields (whichever is extractable from the query):

### Context: {context}
### Query : {query}

### Valid JSON format:
```json
{{
  "source": "<city>",
  "destination": "<city>",
  "date": "dd/mm/yy",
  "days": <integer>,
  "people": <integer>,
  "budget": "Economy" | "Standard" | "Luxury"
}}
### return Just the JSON object noting else(no heading no title no label no explanation no examples just simple JSON object) for efficient further parsing.
"""
    result = query_huggingface(prompt,NER_MODEL_URL,max_tokens=300,temperature=0.5)
    return result

def extract_entities(query):
    prompt = generate_ner_prompt(query)
    result_text = query_huggingface(prompt, NER_MODEL_URL, max_tokens=300, temperature=0.3)
    return result_text

def get_conversational_response(query):
    prompt = generate_chat_prompt(query)
    response = query_huggingface(prompt, CHAT_MODEL_URL, max_tokens=100, temperature=0.5)
    return response

def generate_ner_prompt(query):
    return f"""
### Instruction:
Extract the following fields from the query and respond ONLY as a JSON object:

- source (city)
- destination (city)
- date (format: dd/mm/yy)
- days (integer)
- people (integer)
- budget (Economy | Standard | Luxury)

If any value is missing, return null.

### Query:
"{query}"

### Response:
"""

def generate_chat_prompt(query):
    return f"""
### Instruction:
You are a helpful travel assistant. Provide a natural, conversational response to the user's query about travel planning.

### Query:
"{query}"

### Response:
"""

if __name__ == "__main__":
    # Test entity extraction
    # ner_query = "I want to go from Mumbai to Delhi from 05 February for 6 days with 2 people and a budget is Economy"
    # ner_result = extract_entities(ner_query)
    # print("Entity Extraction Result:")
    # print(json.dumps(ner_result, indent=2))
    
    # Test conversational response
    chat_query = "hi my name is karan."
    chat_response = get_conversational_response(chat_query)
    print("\nConversational Response:")
    print(chat_response)
