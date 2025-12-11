"""
Multi-Provider LLM API Integration

This script demonstrates how to connect to and interact with multiple Large Language Model (LLM) 
providers using Python. It showcases both OpenAI-compatible client libraries and native SDKs 
for various providers.

## Overview

The script covers integration with the following LLM providers:
- OpenAI - GPT models via OpenAI SDK
- Anthropic - Claude models via OpenAI-compatible client and native SDK
- Google Gemini - Gemini models via OpenAI-compatible client and native SDK
- Groq - Fast inference models via OpenAI-compatible client
- DeepSeek - DeepSeek models via OpenAI-compatible client
- Grok - xAI's Grok models via OpenAI-compatible client
- Ollama - Local LLM models via OpenAI-compatible client

## Key Features

1. API Key Management: Loads and validates API keys from environment variables
2. Client Initialization: Sets up clients for multiple providers using OpenAI-compatible endpoints
3. Unified Interface: Demonstrates how to use the OpenAI client library with different base URLs
4. Native SDKs: Shows examples using provider-specific native SDKs (Google Gemini and Anthropic)
5. Practical Example: Tests all providers with a simple joke-telling task

## Usage

1. Set up your API keys in a .env file
2. Run the script: python multi_provider_llm_demo.py
3. Compare responses across different providers

## Requirements

- python-dotenv
- openai
- google-generativeai
- anthropic
- requests
"""

import os
import subprocess
import requests
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from anthropic import Anthropic


# ============================================================================
# API Key Setup
# ============================================================================

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
grok_api_key = os.getenv('GROK_API_KEY')
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key found: {openai_api_key[:8]}...")
else:
    print("OpenAI API Key not set")
    
if anthropic_api_key:
    print(f"Anthropic API Key found: {anthropic_api_key[:7]}...")
else:
    print("Anthropic API Key not set (and this is optional)")

if google_api_key:
    print(f"Google API Key found: {google_api_key[:2]}...")
else:
    print("Google API Key not set (and this is optional)")

if deepseek_api_key:
    print(f"DeepSeek API Key found: {deepseek_api_key[:3]}...")
else:
    print("DeepSeek API Key not set (and this is optional)")

if groq_api_key:
    print(f"Groq API Key found: {groq_api_key[:4]}...")
else:
    print("Groq API Key not set (and this is optional)")

if grok_api_key:
    print(f"Grok API Key found: {grok_api_key[:4]}...")
else:
    print("Grok API Key not set (and this is optional)")

if openrouter_api_key:
    print(f"OpenRouter API Key found: {openrouter_api_key[:3]}...")
else:
    print("OpenRouter API Key not set (and this is optional)")


# ============================================================================
# Client Setup
# ============================================================================

# Connect to OpenAI client library
# A thin wrapper around calls to HTTP endpoints

openai = OpenAI()

# For Gemini, DeepSeek and Groq, we can use the OpenAI python client
# Because Google and DeepSeek have endpoints compatible with OpenAI
# And OpenAI allows you to change the base_url

anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
deepseek_url = "https://api.deepseek.com"
groq_url = "https://api.groq.com/openai/v1"
grok_url = "https://api.x.ai/v1"
openrouter_url = "https://openrouter.ai/api/v1"
ollama_url = "http://localhost:11434/v1"

anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
deepseek = OpenAI(api_key=deepseek_api_key, base_url=deepseek_url)
groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
grok = OpenAI(api_key=grok_api_key, base_url=grok_url)
openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)
ollama = OpenAI(api_key="ollama", base_url=ollama_url)


# ============================================================================
# Test Message
# ============================================================================

tell_a_joke = [
    {"role": "user", "content": "Tell a joke for a student on the journey to becoming an expert in LLM Engineering"},
]


# ============================================================================
# OpenAI
# ============================================================================

print("\n" + "="*80)
print("OpenAI")
print("="*80)
response = openai.chat.completions.create(model="gpt-4.1-mini", messages=tell_a_joke)
print(response.choices[0].message.content)


# ============================================================================
# Anthropic
# ============================================================================

print("\n" + "="*80)
print("Anthropic")
print("="*80)
response = anthropic.chat.completions.create(model="claude-sonnet-4-5-20250929", messages=tell_a_joke)
print(response.choices[0].message.content)


# ============================================================================
# Google Gemini
# ============================================================================

print("\n" + "="*80)
print("Google Gemini")
print("="*80)
response = gemini.chat.completions.create(model="gemini-2.5-pro", messages=tell_a_joke)
print(response.choices[0].message.content)


# ============================================================================
# Groq
# ============================================================================

print("\n" + "="*80)
print("Groq")
print("="*80)
response = groq.chat.completions.create(model="openai/gpt-oss-120b", messages=tell_a_joke)
print(response.choices[0].message.content)


# ============================================================================
# DeepSeek
# ============================================================================

print("\n" + "="*80)
print("DeepSeek")
print("="*80)
response = deepseek.chat.completions.create(model="deepseek-reasoner", messages=tell_a_joke)
print(response.choices[0].message.content)


# ============================================================================
# Grok
# ============================================================================

print("\n" + "="*80)
print("Grok")
print("="*80)
response = grok.chat.completions.create(model="grok-4", messages=tell_a_joke)
print(response.choices[0].message.content)


# ============================================================================
# Ollama
# ============================================================================

print("\n" + "="*80)
print("Ollama")
print("="*80)
requests.get("http://localhost:11434/").content
# Pull the model if not already available
subprocess.run(["ollama", "pull", "llama3.2"], check=False)
response = ollama.chat.completions.create(model="llama3.2", messages=tell_a_joke)
print(response.choices[0].message.content)


# ============================================================================
# Google Gemini (Native SDK)
# ============================================================================

print("\n" + "="*80)
print("Google Gemini (Native SDK)")
print("="*80)
client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash-lite", contents="tell a joke"
)
print(response.text)


# ============================================================================
# Anthropic (Native SDK)
# ============================================================================

print("\n" + "="*80)
print("Anthropic (Native SDK)")
print("="*80)
client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "tell a joke"}],
    max_tokens=100
)
print(response.content[0].text)

