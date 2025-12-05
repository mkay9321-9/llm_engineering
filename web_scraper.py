"""
Web Scraper with LLM Analysis

This script demonstrates how to scrape website content and analyze it using 
OpenAI's API for financial market insights.

Key Concepts:
- Environment variable management with python-dotenv
- Web scraping with BeautifulSoup and requests
- OpenAI Chat API integration
- Prompt engineering for structured outputs
"""

import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import openai
from openai import OpenAI

# ============================================================================
# Environment Variables & API Key Setup
# ============================================================================
# Using python-dotenv to securely load API keys from a .env file.
# This keeps sensitive credentials out of code.

# Load environment variables in a file called .env
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Check the key
if not api_key:
    print("No API key was found!")
else:
    print("API key found and looks good so far!")

# ============================================================================
# Prompt Engineering
# ============================================================================
# System Prompt: Defines the AI's role and behavior (financial research assistant).
# User Prompt: Provides context and instructions for each request.
#              The prefix is combined with website content to form the complete user message.

# Define our system prompt - you can experiment with this later,
# changing the last sentence to 'Respond in markdown in Spanish.'
system_prompt = """
You are a financial research assistant expert at stock market trading. Your job is to analyze the contents of a website, and 
extract information that can signal bullish or bearish trading for the day. 
"""

# Define our user prompt
user_prompt_prefix = """
Here are the contents of a website.

"""

# ============================================================================
# Message Format for OpenAI API
# ============================================================================
# The messages_for() function creates the required message format:
# - system role: Sets AI behavior
# - user role: Contains the actual request with website content
# This follows OpenAI's Chat Completions API structure.

def messages_for(website):
    """Create message format for OpenAI API."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + website}
    ]

# ============================================================================
# Web Scraping with BeautifulSoup
# ============================================================================
# Key techniques:
# - requests.get(): Fetches HTML content (with User-Agent header to avoid blocking)
# - BeautifulSoup: Parses HTML and extracts text
# - decompose(): Removes irrelevant elements (scripts, styles, images)
# - Text extraction: Gets clean text content, limited to 2000 characters

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

def fetch_website_contents(url):
    """Fetch and extract text content from a website."""
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return (title + "\n\n" + text)[:2_000]

# ============================================================================
# OpenAI API Integration
# ============================================================================
# The summarize() function:
# 1. Fetches website content
# 2. Formats messages for the API
# 3. Calls openai.chat.completions.create() with model and messages
# 4. Returns the AI-generated analysis
# Model: gpt-4o-mini (corrected from gpt-4.1-mini)

def summarize(url):
    """Summarize website content using OpenAI API."""
    website = fetch_website_contents(url)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for(website)
    )
    return response.choices[0].message.content

# ============================================================================
# Display Formatting
# ============================================================================
# For command-line execution, we use print() instead of IPython.display.Markdown

def display_summary(url):
    """Fetch summary and print it."""
    summary = summarize(url)
    print(summary)

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    url = "https://www.cnbc.com"
    display_summary(url)

