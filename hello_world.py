"""
Hello World with OpenAI API

This script introduces the basics of interacting with OpenAI's Chat API.

Key Concepts:
- Environment variable management for API keys
- Basic API calls with user messages
- System prompts vs user prompts
- Message format with roles (system and user)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# If you get an error running this script, then please head over to the troubleshooting notebook!

# ============================================================================
# Environment Variables & API Key Setup
# ============================================================================
# Using python-dotenv to securely load API keys from a .env file.
# The OpenAI() client automatically uses the OPENAI_API_KEY environment variable.

# Load environment variables in a file called .env
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key
if not api_key:
    print("No API key was found!")
else:
    print("API key found!")

# ============================================================================
# Basic API Call (User Message Only)
# ============================================================================
# The simplest API call uses only a 'user' role message.
# The model responds based on its default behavior without custom instructions.

# Test Call
message = "Hello, GPT! What is 1+1?"
messages = [{"role": "user", "content": message}]

# Hello World - User Prompt
openai_client = OpenAI()
response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

# ============================================================================
# System Prompts vs User Prompts
# ============================================================================
# System Prompt: Defines the AI's personality, role, and behavior.
#                Sets how the assistant should act across all interactions.
# User Prompt: The actual user request or input. This changes with each conversation turn.
# Key Difference: System prompts shape behavior; user prompts provide the specific request.

# Hello World - System & User Prompt
system_prompt = """
You are a humorous assistant that enjoys in proving customer service,
You will intereact with the user and figure out how to tell a joke that is friendly and non-offensive.
Your job is to make the user feel happy. You will not tell the user that you are an AI assistant.
"""

# User Prompt
user_prompt = """
Hello! Today is a beautiful day!
"""

# ============================================================================
# Complete Message Format
# ============================================================================
# Combining both 'system' and 'user' roles:
# - system: Sets the assistant's behavior (humorous, customer service focused)
# - user: Contains the actual request
# This two-role structure allows you to customize AI behavior while handling dynamic user inputs.

# GPT call
# Note: Using gpt-4o-mini (corrected from gpt-4.1-nano typo in original notebook)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(response.choices[0].message.content)

