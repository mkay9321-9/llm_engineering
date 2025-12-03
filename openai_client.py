"""
OpenAI Client Examples

Using the OpenAI Python client to interact with various LLM providers through a unified interface.
"""

import os
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

# ============================================================================
# Imports
# ============================================================================
# Load environment variables and import OpenAI client.

load_dotenv(override=True)

# ============================================================================
# OpenAI API
# ============================================================================
# Standard OpenAI client using default API endpoint.

def test_openai():
    """Test OpenAI API client."""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("No API key was found!")
        return None
    else:
        print("API key found!")
    
    openai = OpenAI()
    response = openai.chat.completions.create(
        model="gpt-5-nano", 
        messages=[{"role": "user", "content": "Tell me a fun fact"}]
    )
    return response.choices[0].message.content


# ============================================================================
# Google Gemini
# ============================================================================
# Using OpenAI client with Gemini API endpoint via compatibility layer.

def test_gemini():
    """Test Google Gemini API via OpenAI client compatibility layer."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        print("No Google API key was found!")
        return None
    else:
        print("Google API key found!")
    
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
    response = gemini.chat.completions.create(
        model="gemini-2.5-flash-lite", 
        messages=[{"role": "user", "content": "Tell me a fun fact"}]
    )
    return response.choices[0].message.content


# ============================================================================
# Ollama - Llama 3.2
# ============================================================================
# Local Ollama instance with OpenAI-compatible API. Pulls Llama 3.2 model.

def test_ollama_llama():
    """Test Ollama Llama 3.2 model via OpenAI client."""
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
    
    # Pull Llama model using OpenAI client
    print("Pulling llama3.2 model...")
    subprocess.run(["ollama", "pull", "llama3.2"], check=True)
    
    response = ollama.chat.completions.create(
        model="llama3.2", 
        messages=[{"role": "user", "content": "Tell me a fun fact"}]
    )
    return response.choices[0].message.content


# ============================================================================
# Ollama - Deepseek R1
# ============================================================================
# Using Deepseek R1 model via Ollama. Pulls the 1.5B parameter model.

def test_ollama_deepseek():
    """Test Ollama Deepseek R1 model via OpenAI client."""
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')
    
    # Deepseek model using OpenAI client
    print("Pulling deepseek-r1:1.5b model...")
    subprocess.run(["ollama", "pull", "deepseek-r1:1.5b"], check=True)
    
    response = ollama.chat.completions.create(
        model="deepseek-r1:1.5b", 
        messages=[{"role": "user", "content": "Tell me a fun fact"}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Run examples
    print("=" * 60)
    print("OpenAI API Test")
    print("=" * 60)
    result = test_openai()
    if result:
        print(f"Response: {result}\n")
    
    print("=" * 60)
    print("Google Gemini Test")
    print("=" * 60)
    result = test_gemini()
    if result:
        print(f"Response: {result}\n")
    
    print("=" * 60)
    print("Ollama Llama 3.2 Test")
    print("=" * 60)
    try:
        result = test_ollama_llama()
        if result:
            print(f"Response: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    print("=" * 60)
    print("Ollama Deepseek R1 Test")
    print("=" * 60)
    try:
        result = test_ollama_deepseek()
        if result:
            print(f"Response: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")

