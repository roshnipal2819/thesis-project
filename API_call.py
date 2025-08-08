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

# Function to generate a response using few-shot messages with conversation history
def ask_umass_assistant(user_question, conversation_history=None):
    # Check for greetings first (only if no conversation history)
    if not conversation_history or len(conversation_history) == 0:
        user_question_lower = user_question.lower().strip()
        greeting_keywords = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        
        if any(greeting in user_question_lower for greeting in greeting_keywords):
            return "Welcome to EDU Bot, your UMass Dartmouth assistant—how can I help you today?"
    
    # First, try to find a close match in the dataset (only for standalone questions)
    if not conversation_history or len(conversation_history) <= 2:
        matched_answer = find_best_match(user_question, data)
        if matched_answer:
            return matched_answer
    
    try:
        messages = [
            {
                "role": "system",
                "content": "You are EDUBOT, an intelligent voice assistant specifically designed for students at the University of Massachusetts Dartmouth. You provide helpful, accurate, and friendly responses about university-related topics including academics, campus life, student services, and general university information. Always maintain a welcoming and supportive tone appropriate for university students. If you don't have specific information about a topic, guide students to appropriate university resources or departments. Do not use line breaks (`\n`), colons, or bullets. Example: Your campus address is MacLean Campus Center, University of Massachusetts Dartmouth, 285 Old Westport Road, Dartmouth, MA 02747."
            }
        ]

        # Add conversation history if available
        if conversation_history:
            # Add the full conversation history to maintain context
            for message in conversation_history:
                messages.append(message)
        else:
            # Add few-shot examples from your dataset for new conversations
            for pair in data[:50]:  # Reduced to avoid token limits
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
