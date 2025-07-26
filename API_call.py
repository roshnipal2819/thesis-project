import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import difflib

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client (v1.x style)
client = OpenAI(api_key=api_key)

# Load your example conversations from the JSON file
with open("data.json", "r") as f:
    data = json.load(f)

# Function to find the best match in data.json

def find_best_match(user_question, data, threshold=0.85):
    questions = [pair['conversation'][0]['content'] for pair in data]
    matches = difflib.get_close_matches(user_question, questions, n=1, cutoff=threshold)
    if matches:
        match = matches[0]
        for pair in data:
            if pair['conversation'][0]['content'] == match:
                return pair['conversation'][1]['content']
    return None

# Function to generate a response using few-shot messages
def ask_umass_assistant(user_question):
    # First, try to find a close match in the dataset
    matched_answer = find_best_match(user_question, data)
    if matched_answer:
        return matched_answer
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful voice assistant for the University of Massachusetts Dartmouth, College of Engineering. Respond clearly using official university policies."
            }
        ]

        # Add few-shot examples from your dataset
        for pair in data[:50]:
            messages.append({"role": "user", "content": pair['conversation'][0]['content']})
            messages.append({"role": "assistant", "content": pair['conversation'][1]['content']})

        # Append the new user question
        messages.append({"role": "user", "content": user_question})

        # Call OpenAI chat API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error in completion: {str(e)}"

if __name__ == "__main__":
    while True:
        try:
            user_input = input("Ask your question (or type 'exit'): ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            response = ask_umass_assistant(user_input)
            print(f"Assistant: {response}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
